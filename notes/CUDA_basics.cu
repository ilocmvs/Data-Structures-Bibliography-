// vector_add_todo.cu
//
// Practice goal:
//   Implement CUDA vector addition: C[i] = A[i] + B[i]
//
// Build:
//   nvcc -O2 vector_add_todo.cu -o vector_add
//
// Run:
//   ./vector_add

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void vectorAddGridStride(const float *A, const float *B, float *C,
                                    int N) {
  // TODO:
  // Compute global thread id.
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // TODO

  // TODO:
  // Compute total number of threads in the whole grid.
  int stride = blockDim.x * gridDim.x; // TODO

  // TODO:
  // Use grid-stride loop.
  for (int i = tid; i < N; i += stride) {
    // TODO:
    // C[i] = ...
    C[i] = A[i] + B[i];
  }
}
// TODO 1:
// Write a CUDA kernel that computes:
//
//   C[i] = A[i] + B[i]
//
// Requirements:
//   1. Compute the global thread index.
//   2. Check that the index is within bounds.
//   3. Read from A and B.
//   4. Write the result to C.
__global__ void vectorAddKernel(const float *A, const float *B, float *C,
                                int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void reduceSumBlock(const float *x, float *partialSums, int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;

  // TODO:
  // Global thread id.
  int gid = blockIdx.x * blockDim.x + threadIdx.x; // TODO

  // TODO:
  // Store local value into shared memory.
  if (gid < N)
    sdata[tid] = x[gid]; // TODO
  else
    sdata[tid] = 0.0f;

  __syncthreads();

  // TODO:
  // Reduce sdata inside the block.
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (/* TODO */ tid < stride) {
      // TODO:
      // sdata[tid] += ...
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  // TODO:
  // One thread writes this block's partial sum.
  if (/* TODO */ tid == 0) {
    partialSums[blockIdx.x] = sdata[0];
  }
}

auto launchSum = [](const float *d_in, float *d_out, int N, int gridSize,
                    int blockSize, size_t sharedBytes, cudaStream_t stream) {
  reduceSumBlock<<<gridSize, blockSize, sharedBytes, stream>>>(d_in, d_out, N);
};

__global__ void reduceMaxBlock(const float *x, float *partialMax, int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;

  // TODO:
  int gid = blockIdx.x * blockDim.x + threadIdx.x; // global thread id

  if (gid < N)
    sdata[tid] = x[gid]; // TODO
  else
    sdata[tid] = -FLT_MAX;

  __syncthreads();

  // TODO:
  // Shared-memory max reduction.
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (/* TODO */ tid < stride) {
      // TODO:
      // sdata[tid] = fmaxf(...);
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }

  // TODO:
  // Write block max.
  if (/* TODO */) {
    partialMax[blockIdx.x] = sdata[0];
  }
}

auto launchMax = [](const float *d_in, float *d_out, int N, int gridSize,
                    int blockSize, size_t sharedBytes, cudaStream_t stream) {
  reduceMaxBlock<<<gridSize, blockSize, sharedBytes, stream>>>(d_in, d_out, N);
};

template <typename LaunchKernel>
void reduceOperationRecursive(const float *d_x, float *d_result, int N,
                              int blockSize, LaunchKernel launchKernel,
                              cudaStream_t stream = 0) {
  if (N <= 0) {
    cudaMemsetAsync(d_result, 0, sizeof(float), stream);
    return;
  }

  if (N == 1) {
    cudaMemcpyAsync(d_result, d_x, sizeof(float), cudaMemcpyDeviceToDevice,
                    stream);
    return;
  }

  int maxGridSize = (N + blockSize - 1) / blockSize;

  float *d_tempA = nullptr;
  float *d_tempB = nullptr;

  cudaMalloc((void **)&d_tempA, maxGridSize * sizeof(float));
  cudaMalloc((void **)&d_tempB, maxGridSize * sizeof(float));

  const float *d_in = d_x;
  float *d_out = d_tempA;

  int curN = N;

  while (curN > 1) {
    int gridSize = (curN + blockSize - 1) / blockSize;
    size_t sharedBytes = blockSize * sizeof(float);

    launchKernel(d_in, d_out, curN, gridSize, blockSize, sharedBytes, stream);

    curN = gridSize;

    const float *nextIn = d_out;
    d_out = (d_out == d_tempA) ? d_tempB : d_tempA;
    d_in = nextIn;
  }

  cudaMemcpyAsync(d_result, d_in, sizeof(float), cudaMemcpyDeviceToDevice,
                  stream);

  cudaFree(d_tempA);
  cudaFree(d_tempB);
}

__global__ void softmax1DOneBlock(const float *x, float *y, int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;

  // -----------------------------
  // Step 1: find max(x)
  // -----------------------------

  float localMax = -FLT_MAX;

  // TODO:
  // Each thread checks elements tid, tid + blockDim.x, ...
  for (int i = tid; i < N; i += blockDim.x) {
    // TODO:
    // localMax = ...
    localMax = fmaxf(localMax, x[i]);
  }

  // TODO:
  // Store localMax to shared memory.
  sdata[tid] = localMax; // TODO

  __syncthreads();

  // TODO:
  // Reduce to find max.
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (/* TODO */ tid < offset) {
      // TODO
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
    }
    __syncthreads();
  }

  float maxVal = sdata[0];

  // -----------------------------
  // Step 2: compute sum exp(x - max)
  // -----------------------------

  float localSum = 0.0f;

  // TODO:
  // Each thread accumulates part of denominator.
  for (int i = tid; i < N; i += blockDim.x) {
    // TODO:
    // localSum += expf(...);
    localSum += expf(x[i] - maxVal);
  }

  // TODO:
  // Store localSum to shared memory.
  sdata[tid] = localSum; // TODO

  __syncthreads();

  // TODO:
  // Reduce sum.
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (/* TODO */ tid < offset) {
      // TODO
      sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
  }

  float denom = sdata[0];

  // -----------------------------
  // Step 3: normalize
  // -----------------------------

  // TODO:
  // Each thread writes y[i].
  for (int i = tid; i < N; i += blockDim.x) {
    // TODO:
    // y[i] = ...
    y[i] = expf(x[i] - maxVal) / denom;
  }
}

__global__ void histogramAtomicGlobal(const unsigned char *data, int *hist,
                                      int N) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = gid; i < N; i += stride) {
    unsigned char value = data[i];

    // TODO:
    // Safely increment hist[value].
    // Hint: atomicAdd
    atomicAdd(&hist[value], 1);
  }
}

__global__ void histogramAtomicShared(const unsigned char *data, int *hist,
                                      int N) {
  __shared__ int localHist[HIST_BINS];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // TODO:
  // Initialize localHist using block threads.
  for (int bin = tid; bin < HIST_BINS; bin += blockDim.x) {
    // TODO:
    // localHist[bin] = 0;
    localHist[bin] = 0;
  }

  __syncthreads();

  // TODO:
  // Build per-block histogram in shared memory.
  for (int i = gid; i < N; i += stride) {
    unsigned char value = data[i];

    // TODO:
    // atomicAdd to localHist[value]
    atomicAdd(localHist[value], 1);
  }

  __syncthreads();

  // TODO:
  // Merge localHist into global hist.
  for (int bin = tid; bin < HIST_BINS; bin += blockDim.x) {
    // TODO:
    // atomicAdd(&hist[bin], localHist[bin]);
    atomicAdd(&hist[bin], localHist[bin]);
  }
}

__global__ void inclusiveScanBlock(const float *x, float *y, int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO:
  // Load data into shared memory.
  if (gid < N) {
    sdata[tid] = x[gid];
  } else {
    sdata[tid] = 0.0f;
  }

  __syncthreads();

  // TODO:
  // Inclusive scan inside block.
  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    float val = 0.0f;

    if (tid >= offset) {
      // TODO:
      // val = sdata[tid - offset];
      val = sdata[tid - offset];
    }

    __syncthreads();

    // TODO:
    // Add val to sdata[tid].
    if (tid >= offset) {
      // TODO
      sdata[tid] += val;
    }

    __syncthreads();
  }

  // TODO:
  // Write result.
  if (gid < N) {
    y[gid] = sdata[tid];
  }
}

__global__ void layerNormForwardKernel(const float *__restrict__ X,
                                       const float *__restrict__ gamma,
                                       const float *__restrict__ beta,
                                       float *__restrict__ Y, int num_rows,
                                       int hidden_dim, float eps) {
  // X:     [num_rows, hidden_dim]
  // gamma: [hidden_dim]
  // beta:  [hidden_dim]
  // Y:     [num_rows, hidden_dim]
  //
  // One block handles one row.
  // Threads inside the block reduce across hidden_dim.
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row >= num_rows)
    return;

  extern __shared__ float shared[];
  float *s_sum = shared;                // size: blockDim.x
  float *s_sumsq = shared + blockDim.x; // size: blockDim.x

  const float *x_row = X + row * hidden_dim;
  float *y_row = Y + row * hidden_dim;

  float local_sum = 0.0f;
  float local_sumsq = 0.0f;

  // ------------------------------------------------------------
  // TODO 1:
  // Each thread loops over part of the row:
  //
  // for (int col = tid; col < hidden_dim; col += blockDim.x)
  //
  // Accumulate:
  //   local_sum   += x
  //   local_sumsq += x * x
  // ------------------------------------------------------------

  for (int col = tid; col < hidden_dim; col += blockDim.x) {
    // TODO:
    // float v = ...
    // local_sum += ...
    // local_sumsq += ...
    float v = x_row[col];
    local_sum += v;
    local_sumsq += v * v;
  }

  // Store partial sums into shared memory
  s_sum[tid] = local_sum;
  s_sumsq[tid] = local_sumsq;

  __syncthreads();

  // ------------------------------------------------------------
  // TODO 2:
  // Reduce s_sum and s_sumsq across the block.
  //
  // Standard shared-memory reduction:
  //
  // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  //     if (tid < stride) {
  //         s_sum[tid]   += s_sum[tid + stride];
  //         s_sumsq[tid] += s_sumsq[tid + stride];
  //     }
  //     __syncthreads();
  // }
  //
  // Assumption: blockDim.x is power of 2.
  // ------------------------------------------------------------

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    // TODO:
    // if (tid < stride) { ... }
    // __syncthreads();
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sumsq[tid] += s_sumsq[tid + stride];
    }
    __syncthreads();
  }

  // ------------------------------------------------------------
  // TODO 3:
  // Thread 0 computes mean and inverse std.
  //
  // mean = sum / hidden_dim
  // var  = sumsq / hidden_dim - mean * mean
  // inv_std = rsqrtf(var + eps)
  //
  // Store mean and inv_std back into shared memory so all threads
  // can use them.
  // ------------------------------------------------------------

  if (tid == 0) {
    // TODO:
    // float mean = ...
    // float var = ...
    // float inv_std = ...
    //
    // s_sum[0] = mean;
    // s_sumsq[0] = inv_std;
    float mean = s_sum[0] / hidden_dim;
    float var = s_sumsq[0] / hidden_dim - mean * mean;
    var = fmaxf(var, 0.0f);
    float inv_std = rsqrtf(var + eps);

    s_sum[0] = mean;
    s_sumsq[0] = inv_std;
  }

  __syncthreads();

  float mean = s_sum[0];
  float inv_std = s_sumsq[0];

  // ------------------------------------------------------------
  // TODO 4:
  // Normalize the row.
  //
  // y = gamma[col] * (x - mean) * inv_std + beta[col]
  //
  // Again use a strided loop:
  //
  // for (int col = tid; col < hidden_dim; col += blockDim.x)
  // ------------------------------------------------------------

  for (int col = tid; col < hidden_dim; col += blockDim.x) {
    // TODO:
    // float v = ...
    // float g = gamma[col];
    // float b = beta[col];
    // y_row[col] = ...
    float v = x_row[col];
    float g = gamma[col];
    float b = beta[col];
    y_row[col] = g * (v - mean) * inv_std + b;
  }
}

/* CPU verifications */
void vectorAddCPU(const std::vector<float> &A, const std::vector<float> &B,
                  std::vector<float> &C) {
  for (size_t i = 0; i < A.size(); ++i) {
    C[i] = A[i] + B[i];
  }
}

void layerNormForwardCPU(const float *X, const float *gamma, const float *beta,
                         float *Y, int num_rows, int hidden_dim, float eps) {
  for (int row = 0; row < num_rows; ++row) {
    const float *x_row = X + row * hidden_dim;
    float *y_row = Y + row * hidden_dim;

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int col = 0; col < hidden_dim; ++col) {
      float v = x_row[col];
      sum += v;
      sumsq += v * v;
    }

    float mean = sum / hidden_dim;
    float var = sumsq / hidden_dim - mean * mean;
    float inv_std = 1.0f / std::sqrt(var + eps);

    for (int col = 0; col < hidden_dim; ++col) {
      y_row[col] = gamma[col] * (x_row[col] - mean) * inv_std + beta[col];
    }
  }
}

bool checkResult(const std::vector<float> &gpu, const std::vector<float> &cpu,
                 float eps = 1e-5f) {
  if (gpu.size() != cpu.size()) {
    return false;
  }

  for (size_t i = 0; i < gpu.size(); ++i) {
    if (std::fabs(gpu[i] - cpu[i]) > eps) {
      std::cerr << "Mismatch at index " << i << ": GPU = " << gpu[i]
                << ", CPU = " << cpu[i] << std::endl;
      return false;
    }
  }

  return true;
}

int main() {
  const int N = 1 << 20; // about one million elements
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_A(N);
  std::vector<float> h_B(N);
  std::vector<float> h_C(N, 0.0f);
  std::vector<float> h_ref(N, 0.0f);

  // Initialize input data on host.
  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i) * 0.5f;
    h_B[i] = static_cast<float>(i) * 2.0f;
  }

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  // TODO 2:
  // Allocate GPU memory for d_A, d_B, and d_C.
  //
  // Hint:
  //   cudaMalloc((void**)&d_A, bytes);
  //
  // CHECK_CUDA(...);
  CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

  // TODO 3:
  // Copy h_A and h_B from host to device.
  //
  // Hint:
  //   cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
  //
  // CHECK_CUDA(...);
  CHECK_CUDA(cudaMemCpy(d_A, h_A.data(), bytes, cudaMemCpyHostToDevice));
  CHECK_CUDA(cudaMemCpy(d_B, h_B.data(), bytes, cudaMemCpyHostToDevice));

  // TODO 4:
  // Choose a reasonable block size and grid size.
  //
  // Common choice:
  //   blockSize = 256
  //   gridSize = (N + blockSize - 1) / blockSize
  int blockSize = 256;                            // TODO
  int gridSize = (N + blockSize - 1) / blockSize; // TODO

  // TODO 5:
  // Launch the kernel.
  //
  // Hint:
  //   vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

  // TODO 6:
  // Check whether the kernel launch failed.
  //
  // Hint:
  //   cudaGetLastError()
  cudaGetLastError();

  // TODO 7:
  // Synchronize the device before reading results.
  //
  // Hint:
  //   cudaDeviceSynchronize()
  cudaDeviceSynchronize();

  // TODO 8:
  // Copy d_C back from device to host.
  //
  // Hint:
  //   cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
  cudaMemCpy(h_C.data(), d_C, bytes, cudaMemCpyDeviceToHost);

  // Compute CPU reference result.
  vectorAddCPU(h_A, h_B, h_ref);

  // Validate result.
  if (checkResult(h_C, h_ref)) {
    std::cout << "PASS: GPU result matches CPU result." << std::endl;
  } else {
    std::cout << "FAIL: GPU result does not match CPU result." << std::endl;
  }

  // TODO 9:
  // Free GPU memory.
  //
  // Hint:
  //   cudaFree(d_A);
  //   cudaFree(d_B);
  //   cudaFree(d_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}