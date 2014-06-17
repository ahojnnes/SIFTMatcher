// Author: Johannes L. Schoenberger

#include "sift_matcher.h"

void OpenCLProgram::print_device_info(const size_t device_id) {
  const auto& device = devices_[device_id];

  std::string str;

  std::cout << "------------" << std::endl;
  std::cout << "Device ID: " << device_id << std::endl;
  std::cout << "------------" << std::endl;

  device.getInfo(CL_DEVICE_NAME, &str);
  std::cout << "Name: " << str << std::endl;

  device.getInfo(CL_DEVICE_OPENCL_C_VERSION, &str);
  std::cout << "Version: " << str << std::endl;

  int compute_units;
  device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compute_units);
  std::cout << "Max Compute Units: " << compute_units << std::endl;

  size_t size;
  device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
  std::cout << "Local Memory Size: " << size / 1024 << " KB" << std::endl;

  device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);
  std::cout << "Global Memory Size: " << size / (1024 * 1024) << " MB"
            << std::endl;

  device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
  std::cout << "Max Alloc Size: " << size / (1024 * 1024) << " MB" << std::endl;

  device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
  std::cout << "Max Work-group Size: " << size << std::endl;

  std::vector<size_t> dims;
  device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &dims);
  std::cout << "Max Work-item Dims: (";
  for (const size_t& d : dims) {
    std::cout << d << " ";
  }
  std::cout << "\b)" << std::endl;
}

void OpenCLProgram::print_device_info() {
  for (size_t i = 0; i < devices_.size(); ++i) {
    print_device_info(i);
  }
}

void OpenCLProgram::init_(const size_t device_id) {
  try {
    // Create OpenCL context
    context_ = cl::Context(CL_DEVICE_TYPE_GPU);

    // Detect OpenCL devices
    devices_ = context_.getInfo<CL_CONTEXT_DEVICES>();
    device_id_ = device_id;

    // Create OpenCL command queue
    command_queue_ = cl::CommandQueue(context_, devices_[device_id_]);
  }
  OPENCL_CATCH_ERROR
}

void OpenCLProgram::init_kernels_(
    const std::string& source, const std::vector<std::string>& kernel_names) {
  try {
    // Build CL program object
    cl::Program::Sources sources(
        1, std::make_pair(source.c_str(), source.length()));
    program_ = cl::Program(context_, sources);

    program_.build(devices_);

    // Create CL kernel objects
    for (const auto& kernel_name : kernel_names) {
      kernels_[kernel_name] = cl::Kernel(program_, kernel_name.c_str());
    }
  }
  OPENCL_CATCH_ERROR

  const std::string build_log =
      program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices_[device_id_]);
  if (build_log.size() > 0) {
    std::cout << build_log << std::endl;
  }
}

std::string OpenCLProgram::load_kernel_source_(const std::string& path) {
  std::ifstream source_file(path);
  if (source_file.fail()) {
    throw cl::Error(1, "Failed to open OpenCL kernel source file.");
  }
  return std::string(std::istreambuf_iterator<char>(source_file),
                     std::istreambuf_iterator<char>());
}

SIFTMatcher::SIFTMatcher(const size_t device_id, const size_t max_num_images,
                         const size_t max_num_features, const size_t dim)
    : max_num_images_(max_num_images),
      max_num_features_(max_num_features),
      dim_(dim) {
  init_(device_id);

  // Initialize kernels
  const std::string kernel_source =
#include "sift_matcher.cl"
      std::vector<std::string> kernel_names(3);
  kernel_names[0] = "multiply_descriptors";
  kernel_names[1] = "find_row_max";
  kernel_names[2] = "find_col_max";

  init_kernels_(kernel_source, kernel_names);

  // Initialize buffers
  try {
    d_descriptors_.resize(max_num_images);
    d_descriptors_rows_.resize(max_num_images);
    d_descriptors_used_.resize(max_num_images);
    const size_t num_elems = max_num_features * dim;
    for (size_t i = 0; i < max_num_images; ++i) {
      d_descriptors_[i] =
          cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(uint8_t) * num_elems);
      d_descriptors_rows_[i] = 0;
      d_descriptors_used_[i] = false;
    }
    d_dist_matrix_ =
        cl::Buffer(context_, CL_MEM_READ_WRITE,
                   sizeof(float) * max_num_features_ * max_num_features_);
    d_matches12_ =
        cl::Buffer(context_, CL_MEM_WRITE_ONLY, max_num_features * sizeof(int));
    d_matches21_ =
        cl::Buffer(context_, CL_MEM_WRITE_ONLY, max_num_features * sizeof(int));
    h_matches12_ = std::vector<int>(max_num_features);
    h_matches21_ = std::vector<int>(max_num_features);
  }
  OPENCL_CATCH_ERROR
}

bool SIFTMatcher::exists_descriptors(const size_t image_id) {
  return image_to_buffer_.count(image_id);
}

void SIFTMatcher::upload_descriptors(const size_t image_id,
                                     const FeatureDescriptors& descriptors) {
  if (image_to_buffer_.size() >= max_num_images_) {
    throw std::domain_error(
        "Maximum number of descriptors uploaded. Release before uploading new "
        "descriptors.");
  }
  if (descriptors.rows() > max_num_features_) {
    throw std::domain_error(
        "Number of descriptor rows is larger than `max_num_features`.");
  }

  // Find first unused device buffer
  size_t i;
  for (i = 0; i < d_descriptors_used_.size(); ++i) {
    if (!d_descriptors_used_[i]) {
      break;
    }
  }

  const size_t num_bytes = sizeof(uint8_t) * descriptors.size();

  // Upload to device
  try {
    command_queue_.enqueueWriteBuffer(d_descriptors_[i], CL_FALSE, 0, num_bytes,
                                      descriptors.data());
    // Make sure to set the remaining buffer to zero, as matrix multiplication
    // also accesses out of bounds elements
    uint8_t pattern = 0;
    const size_t max_num_bytes = max_num_features_ * dim_ * sizeof(uint8_t);
    command_queue_.enqueueFillBuffer<uint8_t>(
        d_descriptors_[i], pattern, num_bytes, max_num_bytes - num_bytes);
    command_queue_.finish();
  }
  OPENCL_CATCH_ERROR

  image_to_buffer_[image_id] = i;
  d_descriptors_rows_[i] = descriptors.rows();
  d_descriptors_used_[i] = true;
}

void SIFTMatcher::release_descriptors() {
  std::vector<size_t> image_ids;
  for (const auto& d : image_to_buffer_) {
    image_ids.push_back(d.first);
  }
  for (const auto& image_id : image_ids) {
    release_descriptors(image_id);
  }
}

void SIFTMatcher::release_descriptors(const size_t image_id) {
  if (!image_to_buffer_.count(image_id)) {
    throw std::domain_error("Image ID does not exist.");
  }
  const size_t i = image_to_buffer_[image_id];
  image_to_buffer_.erase(image_id);
  d_descriptors_rows_[i] = 0;
  d_descriptors_used_[i] = false;
}

FeatureMatches SIFTMatcher::match(const size_t image_id1,
                                  const size_t image_id2, const float max_ratio,
                                  const float max_dist,
                                  const bool cross_check) {
  if (!image_to_buffer_.count(image_id1) ||
      !image_to_buffer_.count(image_id2)) {
    throw std::domain_error("Image ID does not exist.");
  }

  const size_t i1 = image_to_buffer_[image_id1];
  const size_t i2 = image_to_buffer_[image_id2];
  const size_t rows1 = d_descriptors_rows_[i1];
  const size_t rows2 = d_descriptors_rows_[i2];

  // Make sure everything is uploaded
  try {
    command_queue_.finish();
  }
  OPENCL_CATCH_ERROR

  //////////////////////////////////////////////////////////////////////////////
  // Multiply all pairwise descriptors
  //////////////////////////////////////////////////////////////////////////////

  try {
    auto& kernel = kernels_["multiply_descriptors"];
    kernel.setArg(0, d_descriptors_[i1]);
    kernel.setArg(1, d_descriptors_[i2]);
    kernel.setArg(2, d_dist_matrix_);
    kernel.setArg(3, dim_);
    kernel.setArg(4, rows1);
    kernel.setArg(5, rows2);

    const cl::NDRange global_size(OPENCL_CEIL_SIZE(rows2, MUL_BLOCK_SIZE),
                                  OPENCL_CEIL_SIZE(rows1, MUL_BLOCK_SIZE));
    const cl::NDRange local_size(MUL_BLOCK_SIZE, MUL_BLOCK_SIZE);

    command_queue_.enqueueNDRangeKernel(kernel, cl::NullRange, global_size,
                                        local_size);

    // Make sure we finish before finding maxima
    command_queue_.finish();

    // DEBUG: Download multiplication result
    // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    // mat(rows1, rows2);
    // command_queue_.enqueueReadBuffer(d_dist_matrix_, CL_TRUE, 0,
    // sizeof(float) * rows1 * rows2, mat.data());
    // std::cout << mat.bottomRows(20).rightCols(20) << std::endl;
  }
  OPENCL_CATCH_ERROR

  //////////////////////////////////////////////////////////////////////////////
  // Find maxima
  //////////////////////////////////////////////////////////////////////////////

  // TODO: improve speed for `find_col_max` and `find_row_max` kernel

  // Find row maxima
  try {
    auto& kernel = kernels_["find_row_max"];
    kernel.setArg(0, d_dist_matrix_);
    kernel.setArg(1, d_matches12_);
    kernel.setArg(2, rows1);
    kernel.setArg(3, rows2);
    kernel.setArg(4, max_ratio);
    kernel.setArg(5, max_dist);

    const cl::NDRange global_size(OPENCL_CEIL_SIZE(rows1, FIND_MAX_SIZE));
    const cl::NDRange local_size(FIND_MAX_SIZE);

    command_queue_.enqueueNDRangeKernel(kernel, cl::NullRange, global_size,
                                        local_size);
    command_queue_.enqueueReadBuffer(d_matches12_, CL_FALSE, 0,
                                     rows1 * sizeof(int), h_matches12_.data());
  }
  OPENCL_CATCH_ERROR

  try {
    command_queue_.finish();
  }
  OPENCL_CATCH_ERROR

  if (cross_check) {
    // Find col maxima
    try {
      auto& kernel = kernels_["find_col_max"];
      kernel.setArg(0, d_dist_matrix_);
      kernel.setArg(1, d_matches21_);
      kernel.setArg(2, rows1);
      kernel.setArg(3, rows2);
      kernel.setArg(4, max_ratio);
      kernel.setArg(5, max_dist);

      const cl::NDRange global_size(OPENCL_CEIL_SIZE(rows2, FIND_MAX_SIZE));
      const cl::NDRange local_size(FIND_MAX_SIZE);

      command_queue_.enqueueNDRangeKernel(kernel, cl::NullRange, global_size,
                                          local_size);
      command_queue_.enqueueReadBuffer(
          d_matches21_, CL_FALSE, 0, rows2 * sizeof(int), h_matches21_.data());
    }
    OPENCL_CATCH_ERROR
  }

  // Make sure all computation is finished
  try {
    command_queue_.finish();
  }
  OPENCL_CATCH_ERROR

  //////////////////////////////////////////////////////////////////////////////
  // Cross check
  //////////////////////////////////////////////////////////////////////////////

  FeatureMatches matches(rows1, 2);

  size_t num_valid_matches = 0;
  if (cross_check) {
    for (size_t i = 0; i < rows1; ++i) {
      if (h_matches12_[i] != -1 && i == h_matches21_[h_matches12_[i]]) {
        matches(num_valid_matches, 0) = i;
        matches(num_valid_matches, 1) = h_matches12_[i];
        num_valid_matches += 1;
      }
    }
  } else {
    for (size_t i = 0; i < rows1; ++i) {
      if (h_matches12_[i] != -1) {
        matches(num_valid_matches, 0) = i;
        matches(num_valid_matches, 1) = h_matches12_[i];
        num_valid_matches += 1;
      }
    }
  }

  return matches.topRows(num_valid_matches);
}
