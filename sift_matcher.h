// Author: Johannes L. Schoenberger

#ifndef SIFT_MATCHER_H_
#define SIFT_MATCHER_H_

#include <vector>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <unordered_map>

#include <Eigen/Core>

#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>

#include "sift_matcher.h"

#define OPENCL_CATCH_ERROR                                              \
  catch (cl::Error error) {                                             \
    printf("OpenCL error [%s, line %i]: %s (%d)\n", __FILE__, __LINE__, \
           error.what(), error.err());                                  \
  }

#define OPENCL_CEIL_SIZE(n, w) (int)(n + w - 1) / (int)(w) * (w)

typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptors;
typedef Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> FeatureMatches;

class OpenCLProgram {

 public:
  void print_device_info();
  void print_device_info(const size_t device_id);

 protected:
  void init_(const size_t device_id = 0);
  void init_kernels_(const std::string& source,
                     const std::vector<std::string>& kernel_names);

  static std::string load_kernel_source_(const std::string& path);

  cl::Context context_;
  cl::CommandQueue command_queue_;
  cl::Program program_;
  std::vector<cl::Device> devices_;
  size_t device_id_;
  std::unordered_map<std::string, cl::Kernel> kernels_;
};

class SIFTMatcher : public OpenCLProgram {

 public:
  SIFTMatcher(const size_t device_id = 0, const size_t num_images = 2,
              const size_t max_num_features = 10000, const size_t dim = 128);

  bool exists_descriptors(const size_t image_id);
  void upload_descriptors(const size_t image_id,
                          const FeatureDescriptors& descriptors);
  void release_descriptors(const size_t image_id);
  void release_descriptors();

  FeatureMatches match(const size_t image_id1, const size_t image_id2,
                       const float max_ratio = 0.8, const float max_dist = 0.7,
                       const bool cross_check = true);

 private:
  const static size_t MUL_BLOCK_SIZE = 16;
  const static size_t FIND_MAX_SIZE = 128;

  size_t max_num_images_;
  size_t max_num_features_;
  size_t dim_;

  std::unordered_map<size_t, size_t> image_to_buffer_;
  std::vector<cl::Buffer> d_descriptors_;
  std::vector<bool> d_descriptors_used_;
  std::vector<size_t> d_descriptors_rows_;

  cl::Buffer d_dist_matrix_;
  cl::Buffer d_matches12_;
  cl::Buffer d_matches21_;

  std::vector<int> h_matches12_;
  std::vector<int> h_matches21_;
};

#endif  // SIFT_MATCHER_H_
