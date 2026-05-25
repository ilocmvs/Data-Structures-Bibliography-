// matrix_add_todo.cu
//
// Practice goal:
//   Implement CUDA matrix addition:
//
//      C[row][col] = A[row][col] + B[row][col]
//
// Matrix layout:
//   Row-major flattened 1-D array:
//
//      index = row * cols + col
//
// Build:
//   nvcc -O2 matrix_add_todo.cu -o matrix_add
//
// Run:
//   ./matrix_add

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

__global__ void matrixAdd2D(const float *A, const float *B, float *C, int rows,
                            int cols) {
  // TODO:
  // Map thread/block indices to matrix coordinate.
  int col = blockIdx.x * blockDim.x + threadIdx.x; // TODO
  int row = blockIdx.y * blockDim.y + threadIdx.y; // TODO

  // TODO:
  // Boundary check.
  if (/* TODO */ row < rows && col < cols) {
    // TODO:
    // Convert 2D coordinate to flattened row-major index.
    int idx = row * cols + col; // TODO

    // TODO:
    // C[idx] = ...
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void matmulNaive(const float *A, const float *B, float *C, int M,
                            int K, int N) {
  // TODO:
  // row and col of C.
  int col = blockIdx.x * blockDim.x + threadIdx.x; // TODO
  int row = blockIdx.y * blockDim.y + threadIdx.y; // TODO

  if (/* TODO: row < M && col < N */ row < M && col < N) {
    float sum = 0.0f;

    // TODO:
    // Dot product of A row and B column.
    for (int t = 0; t < K; ++t) {
      // TODO:
      // A[row, t] * B[t, col]
      sum += A[row * K + t] * B[N * t + col];
    }

    // TODO:
    // Store C[row, col].
    C[row * N + col] = sum;
  }
}

__global__ void matmulTiled(const float *A, const float *B, float *C, int M,
                            int K, int N) {
  // One dynamic shared buffer; host passes bytes for As + Bs tiles.
  extern __shared__ float shm[];
  float *As = shm;
  float *Bs = shm + blockDim.x * blockDim.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // TODO:
  // Compute global row and col of C (same mapping as matmulNaive).
  int col = blockIdx.x * blockDim.x + threadIdx.x; // TODO
  int row = blockIdx.y * blockDim.y + threadIdx.y; // TODO

  float sum = 0.0f;

  // TODO:
  // Number of K-tiles.
  int numTiles = (K + blockDim.x - 1) / blockDim.x; // TODO

  for (int tile = 0; tile < numTiles; ++tile) {
    // TODO:
    // Global A coordinate loaded by this thread.
    int aRow = row;                        // TODO
    int aCol = tile * blockDim.x + tx;     // TODO

    // TODO:
    // Global B coordinate loaded by this thread.
    int bRow = tile * blockDim.x + ty;     // TODO
    int bCol = col;                        // TODO

    // TODO:
    // Load A tile into As.
    // If out of bounds, write 0.
    if (/* TODO */ aRow < M && aCol < K) {
      As[ty * blockDim.x + tx] = A[aRow * K + aCol];
    } else {
      As[ty * blockDim.x + tx] = 0.0f;
    }

    // TODO:
    // Load B tile into Bs.
    // If out of bounds, write 0.
    if (/* TODO */ bRow < K && bCol < N) {
      Bs[ty * blockDim.x + tx] = B[bRow * N + bCol];
    } else {
      Bs[ty * blockDim.x + tx] = 0.0f;
    }

    __syncthreads();

    // TODO:
    // Use shared memory to accumulate partial dot product.
    for (int t = 0; t < blockDim.x; ++t) {
      // TODO:
      // sum += As[ty][t] * Bs[t][tx];
      sum += As[ty * blockDim.x + t] * Bs[t * blockDim.x + tx];
    }

    __syncthreads();
  }

  // TODO:
  // Store output if row/col is valid.
  if (/* TODO */ row < M && col < N) {
    // TODO
    C[row * N + col] = sum;
  }
}

__global__ void transposeNaive(const float *in, float *out, int rows,
                               int cols) {
  // TODO:
  // Each thread handles one input element.
  int col = blockIdx.x * blockDim.x + threadIdx.x; // TODO
  int row = blockIdx.y * blockDim.y + threadIdx.y; // TODO

  if (/* TODO */ row < rows && col < cols) {
    // TODO:
    // Read in[row, col]
    // Write out[col, row]
    out[col * rows + row] = in[row * cols + col];
  }
}

__global__ void transposeTiled(const float *in, float *out, int rows,
                               int cols) {
  // +1 padding helps avoid shared memory bank conflicts.
  extern __shared__ float tile[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // TODO:
  // Global coordinates for reading input.
  int x = blockIdx.x * blockDim.x + threadIdx.x; // column in input
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row in input

  // TODO:
  // Load input tile into shared memory.
  if (/* TODO: y < rows && x < cols */ y < rows && x < cols) {
    // TODO:
    // tile[ty][tx] = in[y * cols + x];
    tile[ty * blockDim.x + tx] = in[y * cols + x];
  }

  __syncthreads();

  // TODO:
  // Compute transposed block coordinates.
  // Swap blockIdx.x and blockIdx.y.
  int transposedX = blockIdx.y * blockDim.x + tx; // TODO
  int transposedY = blockIdx.x * blockDim.x + ty; // TODO

  // TODO:
  // Write transposed tile to output.
  if (transposedY < cols && transposedX < rows) {
    // TODO:
    // out[transposedY * rows + transposedX] = tile[tx][ty];
    out[transposedY * rows + transposedX] = tile[tx * blockDim.x + ty];
  }
}

__global__ void conv2d_naive_kernel(const float *input, const float *filter,
                                    float *output, int H, int W, int K,
                                    int padding, int stride) {

  // input:  H x W matrix, row-major
  // filter: K x K matrix, row-major, K is odd
  // output: H x W matrix, row-major
  //

  int H_out = (H + 2 * padding - K) / stride + 1;
  int W_out = (W + 2 * padding - K) / stride + 1;
  // TODO 1:
  // Compute global output row and column handled by this thread.
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO 2:
  // Boundary check: return if this thread is outside output matrix.
  if (/* TODO */ row >= H_out || col >= W_out) {
    return;
  }

  float sum = 0.0f;

  // TODO 3:
  // Loop over filter window.
  for (int fr = 0; fr < K; ++fr) {
    for (int fc = 0; fc < K; ++fc) {

      // TODO 4:
      // Convert filter coordinate to input coordinate.
      int in_r = /* TODO */ row * stride + fr - padding;
      int in_c = /* TODO */ col * stride + fc - padding;

      // TODO 5:
      // Check whether input coordinate is valid.
      if (/* TODO */ 0 <= in_r && in_r < H && in_c < W && in_c >= 0) {

        // TODO 6:
        // Load input value and filter value.
        float x = /* TODO */ input[in_r * W + in_c];
        float w = /* TODO */ filter[fr * K + fc];

        // TODO 7:
        // Accumulate.
        sum += /* TODO */ x * w;
      }
    }
  }

  // TODO 8:
  // Store result.
  output[/* TODO */ row * W_out + col] = sum;
}



/* CPU verifications */
void matrixAddCPU(const std::vector<float> &A, const std::vector<float> &B,
                  std::vector<float> &C, int rows, int cols) {
  C.resize(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int idx = row * cols + col;
      C[static_cast<size_t>(idx)] =
          A[static_cast<size_t>(idx)] + B[static_cast<size_t>(idx)];
    }
  }
}

void matmulCPU(const std::vector<float> &A, const std::vector<float> &B,
               std::vector<float> &C, int M, int K, int N) {
  C.assign(static_cast<size_t>(M) * N, 0.0f);
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float sum = 0.0f;
      for (int t = 0; t < K; ++t) {
        sum += A[static_cast<size_t>(row * K + t)] *
               B[static_cast<size_t>(t * N + col)];
      }
      C[static_cast<size_t>(row * N + col)] = sum;
    }
  }
}

void transposeCPU(const std::vector<float> &in, std::vector<float> &out,
                  int rows, int cols) {
  out.resize(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      out[static_cast<size_t>(col * rows + row)] =
          in[static_cast<size_t>(row * cols + col)];
    }
  }
}

void conv2dCPU(const std::vector<float> &input, const std::vector<float> &filter,
               std::vector<float> &output, int H, int W, int K, int padding,
               int stride) {
  const int H_out = (H + 2 * padding - K) / stride + 1;
  const int W_out = (W + 2 * padding - K) / stride + 1;
  output.assign(static_cast<size_t>(H_out) * W_out, 0.0f);

  for (int row = 0; row < H_out; ++row) {
    for (int col = 0; col < W_out; ++col) {
      float sum = 0.0f;
      for (int fr = 0; fr < K; ++fr) {
        for (int fc = 0; fc < K; ++fc) {
          const int in_r = row * stride + fr - padding;
          const int in_c = col * stride + fc - padding;
          if (0 <= in_r && in_r < H && 0 <= in_c && in_c < W) {
            sum += input[static_cast<size_t>(in_r * W + in_c)] *
                   filter[static_cast<size_t>(fr * K + fc)];
          }
        }
      }
      output[static_cast<size_t>(row * W_out + col)] = sum;
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

void fillMatrixAddInputs(int rows, int cols, std::vector<float> &A,
                         std::vector<float> &B) {
  const int N = rows * cols;
  A.resize(N);
  B.resize(N);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int idx = row * cols + col;
      A[static_cast<size_t>(idx)] = static_cast<float>(row);
      B[static_cast<size_t>(idx)] = static_cast<float>(col);
    }
  }
}

void fillMatmulInputs(int M, int K, int N, std::vector<float> &A,
                      std::vector<float> &B) {
  A.resize(static_cast<size_t>(M) * K);
  B.resize(static_cast<size_t>(K) * N);
  for (int i = 0; i < static_cast<int>(A.size()); ++i) {
    A[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.013f);
  }
  for (int i = 0; i < static_cast<int>(B.size()); ++i) {
    B[static_cast<size_t>(i)] =
        std::cos(static_cast<float>(i) * 0.017f);
  }
}

void fillTransposeInputs(int rows, int cols, std::vector<float> &in) {
  in.resize(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      in[static_cast<size_t>(row * cols + col)] =
          static_cast<float>(row * 1000 + col);
    }
  }
}

void fillConv2dInputs(int H, int W, int K, std::vector<float> &input,
                      std::vector<float> &filter) {
  input.resize(static_cast<size_t>(H) * W);
  filter.resize(static_cast<size_t>(K) * K);
  for (int i = 0; i < static_cast<int>(input.size()); ++i) {
    input[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.11f);
  }
  for (int i = 0; i < static_cast<int>(filter.size()); ++i) {
    filter[static_cast<size_t>(i)] =
        (i % 2 == 0) ? 0.25f : -0.1f;
  }
}

// matmulTiled uses two extern __shared__ tiles (As and Bs).
static size_t matmulTiledSharedBytes(const dim3 &block) {
  const size_t tileElems =
      static_cast<size_t>(block.x) * static_cast<size_t>(block.y);
  return 2 * tileElems * sizeof(float);
}

// transposeTiled uses one extern __shared__ tile buffer.
static size_t transposeTiledSharedBytes(const dim3 &block) {
  const size_t tileElems =
      static_cast<size_t>(block.x) * static_cast<size_t>(block.y);
  return tileElems * sizeof(float);
}

bool verifyMatrixAdd2D() {
  const int rows = 512;
  const int cols = 512;
  const int N = rows * cols;
  const size_t bytes = static_cast<size_t>(N) * sizeof(float);

  std::vector<float> h_A, h_B, h_C(N, 0.0f), h_ref;
  fillMatrixAddInputs(rows, cols, h_A, h_B);
  matrixAddCPU(h_A, h_B, h_ref, rows, cols);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));
  CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 blockSize(16, 16);
  const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                      (rows + blockSize.y - 1) / blockSize.y);
  matrixAdd2D<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return checkResult(h_C, h_ref, 1e-5f, "matrixAdd2D");
}

bool verifyMatmulNaive() {
  const int M = 128;
  const int K = 64;
  const int N = 96;

  std::vector<float> h_A, h_B, h_C, h_ref;
  fillMatmulInputs(M, K, N, h_A, h_B);
  matmulCPU(h_A, h_B, h_ref, M, K, N);
  h_C.assign(static_cast<size_t>(M) * N, 0.0f);

  const size_t bytesA = h_A.size() * sizeof(float);
  const size_t bytesB = h_B.size() * sizeof(float);
  const size_t bytesC = h_C.size() * sizeof(float);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_A, bytesA));
  CHECK_CUDA(cudaMalloc((void **)&d_B, bytesB));
  CHECK_CUDA(cudaMalloc((void **)&d_C, bytesC));
  CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

  const dim3 blockSize(16, 16);
  const dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);
  matmulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return checkResult(h_C, h_ref, 1e-3f, "matmulNaive");
}

bool verifyMatmulTiled() {
  const int M = 128;
  const int K = 64;
  const int N = 96;

  std::vector<float> h_A, h_B, h_C, h_ref;
  fillMatmulInputs(M, K, N, h_A, h_B);
  matmulCPU(h_A, h_B, h_ref, M, K, N);
  h_C.assign(static_cast<size_t>(M) * N, 0.0f);

  const size_t bytesA = h_A.size() * sizeof(float);
  const size_t bytesB = h_B.size() * sizeof(float);
  const size_t bytesC = h_C.size() * sizeof(float);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_A, bytesA));
  CHECK_CUDA(cudaMalloc((void **)&d_B, bytesB));
  CHECK_CUDA(cudaMalloc((void **)&d_C, bytesC));
  CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

  const dim3 blockSize(16, 16);
  const dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);
  const size_t sharedBytes = matmulTiledSharedBytes(blockSize);
  matmulTiled<<<gridSize, blockSize, sharedBytes>>>(d_A, d_B, d_C, M, K, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return checkResult(h_C, h_ref, 1e-3f, "matmulTiled");
}

bool verifyTransposeNaive() {
  const int rows = 256;
  const int cols = 192;
  const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

  std::vector<float> h_in, h_out, h_ref;
  fillTransposeInputs(rows, cols, h_in);
  transposeCPU(h_in, h_ref, rows, cols);
  h_out.assign(h_in.size(), 0.0f);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));
  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 blockSize(16, 16);
  const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                      (rows + blockSize.y - 1) / blockSize.y);
  transposeNaive<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_in);
  cudaFree(d_out);
  return checkResult(h_out, h_ref, 1e-5f, "transposeNaive");
}

bool verifyTransposeTiled() {
  const int rows = 256;
  const int cols = 192;
  const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

  std::vector<float> h_in, h_out, h_ref;
  fillTransposeInputs(rows, cols, h_in);
  transposeCPU(h_in, h_ref, rows, cols);
  h_out.assign(h_in.size(), 0.0f);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));
  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 blockSize(16, 16);
  const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                      (rows + blockSize.y - 1) / blockSize.y);
  const size_t sharedBytes = transposeTiledSharedBytes(blockSize);
  transposeTiled<<<gridSize, blockSize, sharedBytes>>>(d_in, d_out, rows, cols);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_in);
  cudaFree(d_out);
  return checkResult(h_out, h_ref, 1e-5f, "transposeTiled");
}

bool verifyConv2dNaive() {
  const int H = 64;
  const int W = 64;
  const int K = 3;
  const int padding = 1;
  const int stride = 1;
  const int H_out = (H + 2 * padding - K) / stride + 1;
  const int W_out = (W + 2 * padding - K) / stride + 1;

  std::vector<float> h_input, h_filter, h_output, h_ref;
  fillConv2dInputs(H, W, K, h_input, h_filter);
  conv2dCPU(h_input, h_filter, h_ref, H, W, K, padding, stride);
  h_output.assign(static_cast<size_t>(H_out) * W_out, 0.0f);

  const size_t inputBytes = h_input.size() * sizeof(float);
  const size_t filterBytes = h_filter.size() * sizeof(float);
  const size_t outputBytes = h_output.size() * sizeof(float);

  float *d_input = nullptr;
  float *d_filter = nullptr;
  float *d_output = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_input, inputBytes));
  CHECK_CUDA(cudaMalloc((void **)&d_filter, filterBytes));
  CHECK_CUDA(cudaMalloc((void **)&d_output, outputBytes));
  CHECK_CUDA(
      cudaMemcpy(d_input, h_input.data(), inputBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_filter, h_filter.data(), filterBytes,
                        cudaMemcpyHostToDevice));

  const dim3 blockSize(16, 16);
  const dim3 gridSize((W_out + blockSize.x - 1) / blockSize.x,
                      (H_out + blockSize.y - 1) / blockSize.y);
  conv2d_naive_kernel<<<gridSize, blockSize>>>(d_input, d_filter, d_output, H,
                                                W, K, padding, stride);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, outputBytes,
                        cudaMemcpyDeviceToHost));

  cudaFree(d_input);
  cudaFree(d_filter);
  cudaFree(d_output);
  return checkResult(h_output, h_ref, 1e-4f, "conv2d_naive");
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

  report("matrixAdd2D", verifyMatrixAdd2D());
  report("matmulNaive", verifyMatmulNaive());
  report("matmulTiled", verifyMatmulTiled());
  report("transposeNaive", verifyTransposeNaive());
  report("transposeTiled", verifyTransposeTiled());
  report("conv2d_naive", verifyConv2dNaive());

  if (failures == 0) {
    std::cout << "All kernel checks passed." << std::endl;
    return 0;
  }

  return 1;
}