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

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <iomanip>

constexpr int HIST_BINS = 256;

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
  if (tid == 0) {
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

//warp-level reduction
__inline__ __device__ float warpReduceSum(float val) {
  unsigned mask = 0xffffffff;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
  }

  return val;
}

__inline__ __device__ float blockReduceSum(float val) {
  // Maximum 1024 threads per block -> 1024 / 32 = 32 warps
  __shared__ float warpSums[32];

  int tid  = threadIdx.x;
  int lane = tid % warpSize;
  int wid  = tid / warpSize;

  // First reduce inside each warp
  val = warpReduceSum(val);

  // Lane 0 of each warp writes its warp result
  if (lane == 0) {
      warpSums[wid] = val;
  }

  __syncthreads();

  // First warp reduces all warp-level partial sums
  float blockSum = 0.0f;

  int numWarps = (blockDim.x + warpSize - 1) / warpSize;

  if (wid == 0) {
      blockSum = (lane < numWarps) ? warpSums[lane] : 0.0f;
      blockSum = warpReduceSum(blockSum);
  }

  return blockSum;
}

__global__ void reducePass1(const float* input, float* partial, int N) {
  float sum = 0.0f;

  int tid = threadIdx.x;
  int blockStart = blockIdx.x * blockDim.x * 2;

  int i = blockStart + tid;

  // Load two elements per thread for better memory bandwidth usage
  if (i < N) {
      sum += input[i];
  }

  if (i + blockDim.x < N) {
      sum += input[i + blockDim.x];
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x == 0) {
      partial[blockIdx.x] = sum;
  }
}

__global__ void reducePass2(const float* partial, float* output, int numPartials) {
  float sum = 0.0f;

  for (int i = threadIdx.x; i < numPartials; i += blockDim.x) {
      sum += partial[i];
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x == 0) {
      *output = sum;
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
    atomicAdd(&localHist[value], 1);
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

float reduceSumCPU(const std::vector<float> &x) {
  float sum = 0.0f;
  for (float v : x) {
    sum += v;
  }
  return sum;
}

float reduceMaxCPU(const std::vector<float> &x) {
  float mx = -FLT_MAX;
  for (float v : x) {
    mx = std::fmax(mx, v);
  }
  return mx;
}

void softmax1DCPU(const std::vector<float> &x, std::vector<float> &y) {
  const int N = static_cast<int>(x.size());
  y.resize(N);

  float maxVal = reduceMaxCPU(x);
  float denom = 0.0f;
  for (int i = 0; i < N; ++i) {
    denom += std::exp(x[i] - maxVal);
  }

  for (int i = 0; i < N; ++i) {
    y[i] = std::exp(x[i] - maxVal) / denom;
  }
}

void histogramCPU(const std::vector<unsigned char> &data, std::vector<int> &hist,
                  int bins) {
  hist.assign(bins, 0);
  for (unsigned char value : data) {
    ++hist[value];
  }
}

void inclusiveScanBlockCPU(const std::vector<float> &x, std::vector<float> &y,
                           int blockSize) {
  const int N = static_cast<int>(x.size());
  y.assign(N, 0.0f);

  const int numBlocks = (N + blockSize - 1) / blockSize;
  std::vector<float> block(static_cast<size_t>(blockSize));

  for (int b = 0; b < numBlocks; ++b) {
    std::fill(block.begin(), block.end(), 0.0f);
    for (int tid = 0; tid < blockSize; ++tid) {
      const int gid = b * blockSize + tid;
      if (gid < N) {
        block[static_cast<size_t>(tid)] = x[static_cast<size_t>(gid)];
      }
    }

    for (int offset = 1; offset < blockSize; offset *= 2) {
      std::vector<float> prev = block;
      for (int tid = 0; tid < blockSize; ++tid) {
        if (tid >= offset) {
          block[static_cast<size_t>(tid)] +=
              prev[static_cast<size_t>(tid - offset)];
        }
      }
    }

    for (int tid = 0; tid < blockSize; ++tid) {
      const int gid = b * blockSize + tid;
      if (gid < N) {
        y[static_cast<size_t>(gid)] = block[static_cast<size_t>(tid)];
      }
    }
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
      const float v = x_row[col];
      sum += v;
      sumsq += v * v;
    }

    const float mean = sum / hidden_dim;
    float var = sumsq / hidden_dim - mean * mean;
    var = std::fmax(var, 0.0f);
    const float inv_std = 1.0f / std::sqrt(var + eps);

    for (int col = 0; col < hidden_dim; ++col) {
      y_row[col] =
          gamma[col] * (x_row[col] - mean) * inv_std + beta[col];
    }
  }
}

bool checkResult(const std::vector<float> &gpu, const std::vector<float> &cpu,
                 float tol = 1e-5f, const char *label = "result") {
  if (gpu.size() != cpu.size()) {
    std::cerr << label << ": size mismatch (GPU=" << gpu.size()
              << ", CPU=" << cpu.size() << ")" << std::endl;
    return false;
  }

  for (size_t i = 0; i < gpu.size(); ++i) {
    if (std::fabs(gpu[i] - cpu[i]) > tol) {
      std::cerr << label << ": mismatch at index " << i << " GPU=" << gpu[i]
                << " CPU=" << cpu[i] << std::endl;
      return false;
    }
  }

  return true;
}

bool checkScalar(float gpu, float cpu, float tol = 1e-3f,
                 const char *label = "scalar") {
  if (std::fabs(gpu - cpu) > tol) {
    std::cerr << std::setprecision(10) << label << ": GPU=" << gpu << " CPU=" << cpu << std::endl;
    return false;
  }
  return true;
}

bool checkHistogram(const std::vector<int> &gpu, const std::vector<int> &cpu,
                    const char *label = "histogram") {
  if (gpu.size() != cpu.size()) {
    std::cerr << label << ": size mismatch" << std::endl;
    return false;
  }

  for (size_t i = 0; i < gpu.size(); ++i) {
    if (gpu[i] != cpu[i]) {
      std::cerr << label << ": mismatch at bin " << i << " GPU=" << gpu[i]
                << " CPU=" << cpu[i] << std::endl;
      return false;
    }
  }

  return true;
}

void fillVectorAddInputs(int N, std::vector<float> &A, std::vector<float> &B) {
  A.resize(N);
  B.resize(N);
  for (int i = 0; i < N; ++i) {
    A[static_cast<size_t>(i)] = static_cast<float>(i) * 0.5f;
    B[static_cast<size_t>(i)] = static_cast<float>(i) * 2.0f;
  }
}

bool verifyVectorAddKernel(const char *name,
                           void (*launch)(float *, float *, float *, int, int,
                                          int)) {
  const int N = 1 << 16;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_A, h_B, h_C(N, 0.0f), h_ref(N, 0.0f);
  fillVectorAddInputs(N, h_A, h_B);
  vectorAddCPU(h_A, h_B, h_ref);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));
  CHECK_CUDA(
      cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;
  launch(d_A, d_B, d_C, N, gridSize, blockSize);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  const bool ok = checkResult(h_C, h_ref, 1e-5f, name);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return ok;
}

void launchVectorAddKernel(float *d_A, float *d_B, float *d_C, int N,
                           int gridSize, int blockSize) {
  vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
}

void launchVectorAddGridStride(float *d_A, float *d_B, float *d_C, int N,
                               int gridSize, int blockSize) {
  vectorAddGridStride<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
}

bool verifyReduceSum() {
  const int N = 1 << 18;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_x(N);
  for (int i = 0; i < N; ++i) {
    h_x[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.001f);
  }

  float *d_x = nullptr;
  float *d_out = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_x, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_out, sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  const int blockSize = 256;
  reduceOperationRecursive(d_x, d_out, N, blockSize, launchSum);

  float gpuSum = 0.0f;
  CHECK_CUDA(cudaMemcpy(&gpuSum, d_out, sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float cpuSum = reduceSumCPU(h_x);

  cudaFree(d_x);
  cudaFree(d_out);
  return checkScalar(gpuSum, cpuSum, 1e-2f, "reduceSum");
}

void launchReduceSumTwoPass(const float *d_in, float *d_out, int N,
                            int blockSize = 256) {
  if (N <= 0) {
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
    return;
  }

  if (N == 1) {
    CHECK_CUDA(cudaMemcpy(d_out, d_in, sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return;
  }

  const int gridSize1 = (N + 2 * blockSize - 1) / (2 * blockSize);
  const size_t partialBytes = static_cast<size_t>(gridSize1) * sizeof(float);

  float *d_partial = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_partial, partialBytes));

  reducePass1<<<gridSize1, blockSize>>>(d_in, d_partial, N);
  reducePass2<<<1, blockSize>>>(d_partial, d_out, gridSize1);

  cudaFree(d_partial);
}

bool verifyReduceSumTwoPass() {
  const int N = 1 << 18;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_x(N);
  for (int i = 0; i < N; ++i) {
    h_x[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.001f);
  }

  float *d_x = nullptr;
  float *d_out = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_x, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_out, sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  const int blockSize = 256;
  launchReduceSumTwoPass(d_x, d_out, N, blockSize);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  float gpuSum = 0.0f;
  CHECK_CUDA(cudaMemcpy(&gpuSum, d_out, sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float cpuSum = reduceSumCPU(h_x);

  cudaFree(d_x);
  cudaFree(d_out);
  return checkScalar(gpuSum, cpuSum, 1e-2f, "reduceSumTwoPass");
}

bool verifyReduceMax() {
  const int N = 1 << 18;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_x(N);
  for (int i = 0; i < N; ++i) {
    h_x[static_cast<size_t>(i)] =
        std::cos(static_cast<float>(i) * 0.002f);
  }

  float *d_x = nullptr;
  float *d_out = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_x, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_out, sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  const int blockSize = 256;
  reduceOperationRecursive(d_x, d_out, N, blockSize, launchMax);

  float gpuMax = 0.0f;
  CHECK_CUDA(cudaMemcpy(&gpuMax, d_out, sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float cpuMax = reduceMaxCPU(h_x);

  cudaFree(d_x);
  cudaFree(d_out);
  return checkScalar(gpuMax, cpuMax, 1e-5f, "reduceMax");
}

bool verifySoftmax1D() {
  const int N = 512;
  const int blockSize = 512;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_x(N), h_y(N, 0.0f), h_ref(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    h_x[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.01f);
  }
  softmax1DCPU(h_x, h_ref);

  float *d_x = nullptr;
  float *d_y = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_x, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_y, bytes));
  CHECK_CUDA(
      cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  const size_t sharedBytes = blockSize * sizeof(float);
  softmax1DOneBlock<<<1, blockSize, sharedBytes>>>(d_x, d_y, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_x);
  cudaFree(d_y);
  return checkResult(h_y, h_ref, 1e-5f, "softmax1D");
}

bool verifyHistogram(bool useShared) {
  const int N = 1 << 20;
  const size_t dataBytes = N * sizeof(unsigned char);
  const size_t histBytes = HIST_BINS * sizeof(int);

  std::vector<unsigned char> h_data(N);
  for (int i = 0; i < N; ++i) {
    h_data[static_cast<size_t>(i)] =
        static_cast<unsigned char>(i % HIST_BINS);
  }

  std::vector<int> h_hist(HIST_BINS, 0), h_ref(HIST_BINS, 0);
  histogramCPU(h_data, h_ref, HIST_BINS);

  unsigned char *d_data = nullptr;
  int *d_hist = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_data, dataBytes));
  CHECK_CUDA(cudaMalloc((void **)&d_hist, histBytes));
  CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), dataBytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_hist, 0, histBytes));

  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;
  if (useShared) {
    histogramAtomicShared<<<gridSize, blockSize>>>(d_data, d_hist, N);
  } else {
    histogramAtomicGlobal<<<gridSize, blockSize>>>(d_data, d_hist, N);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_hist.data(), d_hist, histBytes,
                        cudaMemcpyDeviceToHost));

  cudaFree(d_data);
  cudaFree(d_hist);
  return checkHistogram(h_hist, h_ref,
                        useShared ? "histogramShared" : "histogramGlobal");
}

bool verifyInclusiveScanBlock() {
  const int N = 4096;
  const int blockSize = 256;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_x(N), h_y(N, 0.0f), h_ref(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    h_x[static_cast<size_t>(i)] = static_cast<float>((i % 17) - 8);
  }
  inclusiveScanBlockCPU(h_x, h_ref, blockSize);

  float *d_x = nullptr;
  float *d_y = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_x, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_y, bytes));
  CHECK_CUDA(
      cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  const int gridSize = (N + blockSize - 1) / blockSize;
  const size_t sharedBytes = blockSize * sizeof(float);
  inclusiveScanBlock<<<gridSize, blockSize, sharedBytes>>>(d_x, d_y, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_x);
  cudaFree(d_y);
  return checkResult(h_y, h_ref, 1e-5f, "inclusiveScanBlock");
}

bool verifyLayerNorm() {
  const int num_rows = 32;
  const int hidden_dim = 128;
  const int blockSize = 128;
  const float eps = 1e-5f;
  const size_t matBytes =
      static_cast<size_t>(num_rows) * hidden_dim * sizeof(float);
  const size_t vecBytes = hidden_dim * sizeof(float);

  std::vector<float> h_X(static_cast<size_t>(num_rows) * hidden_dim);
  std::vector<float> h_gamma(hidden_dim, 1.0f);
  std::vector<float> h_beta(hidden_dim, 0.0f);
  std::vector<float> h_Y(h_X.size(), 0.0f);
  std::vector<float> h_ref(h_X.size(), 0.0f);

  for (int i = 0; i < static_cast<int>(h_X.size()); ++i) {
    h_X[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.03f);
  }
  for (int i = 0; i < hidden_dim; ++i) {
    h_gamma[static_cast<size_t>(i)] = 0.5f + 0.01f * i;
    h_beta[static_cast<size_t>(i)] = -0.1f + 0.005f * i;
  }

  layerNormForwardCPU(h_X.data(), h_gamma.data(), h_beta.data(),
                      h_ref.data(), num_rows, hidden_dim, eps);

  float *d_X = nullptr;
  float *d_gamma = nullptr;
  float *d_beta = nullptr;
  float *d_Y = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_X, matBytes));
  CHECK_CUDA(cudaMalloc((void **)&d_gamma, vecBytes));
  CHECK_CUDA(cudaMalloc((void **)&d_beta, vecBytes));
  CHECK_CUDA(cudaMalloc((void **)&d_Y, matBytes));
  CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), matBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_gamma, h_gamma.data(), vecBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_beta, h_beta.data(), vecBytes, cudaMemcpyHostToDevice));

  const size_t sharedBytes = 2 * blockSize * sizeof(float);
  layerNormForwardKernel<<<num_rows, blockSize, sharedBytes>>>(
      d_X, d_gamma, d_beta, d_Y, num_rows, hidden_dim, eps);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_Y.data(), d_Y, matBytes, cudaMemcpyDeviceToHost));

  cudaFree(d_X);
  cudaFree(d_gamma);
  cudaFree(d_beta);
  cudaFree(d_Y);
  return checkResult(h_Y, h_ref, 1e-4f, "layerNorm");
}

int main() {
  int failures = 0;

  auto report = [&](const char *name, bool ok) {
    if (ok) {
      std::cout << "PASS: " << name << std::endl;
    } else {
      std::cout << "FAIL: " << name << std::endl;
      ++failures;
    }
  };

  report("vectorAddKernel",
         verifyVectorAddKernel("vectorAddKernel", launchVectorAddKernel));
  report("vectorAddGridStride",
         verifyVectorAddKernel("vectorAddGridStride",
                               launchVectorAddGridStride));
  report("reduceSum", verifyReduceSum());
  report("reduceSumTwoPass", verifyReduceSumTwoPass());
  report("reduceMax", verifyReduceMax());
  report("softmax1D", verifySoftmax1D());
  report("histogramAtomicGlobal", verifyHistogram(false));
  report("histogramAtomicShared", verifyHistogram(true));
  report("inclusiveScanBlock", verifyInclusiveScanBlock());
  report("layerNormForward", verifyLayerNorm());

  if (failures == 0) {
    std::cout << "All kernel checks passed." << std::endl;
    return 0;
  }

  std::cout << failures << " kernel check(s) failed." << std::endl;
  return 1;
}