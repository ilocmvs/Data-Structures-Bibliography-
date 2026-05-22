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
    if (/* TODO */tid < offset) {
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
    if (/* TODO */tid < offset) {
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







void vectorAddCPU(const std::vector<float> &A, const std::vector<float> &B,
                  std::vector<float> &C) {
  for (size_t i = 0; i < A.size(); ++i) {
    C[i] = A[i] + B[i];
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