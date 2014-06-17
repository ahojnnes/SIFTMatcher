#ifndef OPENCL_STRINGIFY
#define OPENCL_STRINGIFY(S) #S
#define OPENCL_XSTRINGIFY(S) OPENCL_STRINGIFY(S)
#endif

#define BLOCK_SIZE 16

OPENCL_XSTRINGIFY(
  /*
    Given two descriptors A and B, in row-major order, calculate

             C = A * B^T

    where A_[A_rows x A_cols],
          B_[B_rows x B_cols],
          C_[C_rows x C_cols]
    and A_cols == B_cols == dim,
        C_rows == A_rows,
        C_cols == B_rows.

    Note, that B is assumed to be passed as B and not B^T. THe transposition
    is automatically done inside the kernel.
   */
  __kernel void multiply_descriptors(const __global uchar* A,
                                     const __global uchar* B,
                                     __global float* C,
                                     const int dim,
                                     const int C_rows,
                                     const int C_cols) {
    // Block index
    const int bx = get_group_id(0);
    const int by = get_group_id(1);

    // Thread index
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed by the block
    const int a0 = dim * BLOCK_SIZE * by;
    // Index of the last sub-matrix of A processed by the block
    const int a1 = a0 + dim - 1;
    // Step size used to iterate through the sub-matrices of A
    const int a_step = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    const int b0 = dim * BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    const int b_step = BLOCK_SIZE;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B required to compute the
    // block sub-matrix
    for (int a=a0, b=b0; a<=a1; a+=a_step, b+=b_step) {
      // Shared memory for work-group
      __local float As[BLOCK_SIZE][BLOCK_SIZE];
      __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

      // Load the matrices from device memory to shared memory;
      // each thread loads one element of each matrix
      As[ty][tx] = (float)A[a + dim * ty + tx];
      Bs[tx][ty] = (float)B[b + dim * ty + tx];

      // Synchronize to make sure the matrices are loaded
      barrier(CLK_LOCAL_MEM_FENCE);

      // Multiply the two matrices together;
      // each thread computes one element of the block sub-matrix
      for (int k=0; k<BLOCK_SIZE; ++k) {
        Csub += As[ty][k] * Bs[k][tx];
      }

      // Synchronize to make sure that the preceding computation is done
      // before loading two new sub-matrices of A and B in the next
      // iteration
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element;
    // avoid duplicate write, if global work size does not match matrix size
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    if (cx < C_cols && cy < C_rows) {
      C[cy * C_cols + cx] = Csub;
    }
  }


  inline float normalize_dist_(const float dist) {
    // Assuming the descriptor values are in range [0, 255] and the original
    // SIFT values are in range [0, 0.5], normalize distance by
    //      1 / 512^2 = 0.000003814697265625f
    return acos(min(1.0f, dist * 0.000003814697265625f));
  }


  // TODO: improve speed for `find_col_max` and `find_row_max` kernel


  __kernel void find_col_max(const __global float* dist_matrix,
                             __global int* matches,
                             const int rows1,
                             const int rows2,
                             const float max_ratio,
                             const float max_dist) {
    const int idx2 = get_global_id(0);
    if (idx2 >= rows2) {
      return;
    }

    float max_dist1 = 0;
    float max_dist2 = 0;
    int max_idx = -1;

    // Find maximum distance in column idx2
    for (int idx1=0; idx1<rows1; ++idx1) {
      const float dist = dist_matrix[idx1 * rows2 + idx2];
      if (dist > max_dist1) {
        max_dist2 = max_dist1;
        max_dist1 = dist;
        max_idx = idx1;
      } else if (dist > max_dist2) {
        max_dist2 = dist;
      }
    }

    max_dist1 = normalize_dist_(max_dist1);
    max_dist2 = normalize_dist_(max_dist2);

    // Ratio test
    if (max_dist1 < max_dist && max_dist1 < max_ratio * max_dist2) {
      matches[idx2] = max_idx;
    } else {
      matches[idx2] = -1;
    }
  }


  __kernel void find_row_max(const __global float* dist_matrix,
                             __global int* matches,
                             const int rows1,
                             const int rows2,
                             const float max_ratio,
                             const float max_dist) {
    const int idx1 = get_global_id(0);
    if (idx1 >= rows1) {
      return;
    }

    float max_dist1 = 0;
    float max_dist2 = 0;
    int max_idx = -1;

    // Find maximum distance in row idx1
    for (int idx2=0; idx2<rows2; ++idx2) {
      const float dist = dist_matrix[idx1 * rows2 + idx2];
      if (dist > max_dist1) {
        max_dist2 = max_dist1;
        max_dist1 = dist;
        max_idx = idx2;
      } else if (dist > max_dist2) {
        max_dist2 = dist;
      }
    }

    max_dist1 = normalize_dist_(max_dist1);
    max_dist2 = normalize_dist_(max_dist2);

    // Ratio test
    if (max_dist1 < max_dist && max_dist1 < max_ratio * max_dist2) {
      matches[idx1] = max_idx;
    } else {
      matches[idx1] = -1;
    }
  }
);

#undef BLOCK_SIZE
