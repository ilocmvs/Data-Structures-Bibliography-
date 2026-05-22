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

#define TILE 16

__global__ void matmulTiled(const float *A, const float *B, float *C, int M,
                            int K, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // TODO:
  // Compute global row and col of C.
  int row = blockIdx.x * blockDim.x + threadIdx.x; // TODO
  int col = blockIdx.y * blockDim.y + threadIdx.y; // TODO

  float sum = 0.0f;

  // TODO:
  // Number of K-tiles.
  int numTiles = (K + TILE - 1) / TILE; // TODO

  for (int tile = 0; tile < numTiles; ++tile) {
    // TODO:
    // Global A coordinate loaded by this thread.
    int aRow = row;              // TODO
    int aCol = ty + tile * TILE; // TODO

    // TODO:
    // Global B coordinate loaded by this thread.
    int bRow = tile * TILE + tx; // TODO
    int bCol = col;              // TODO

    // TODO:
    // Load A tile into As.
    // If out of bounds, write 0.
    if (/* TODO */ aRow < M && aCol < K) {
      As[ty][tx] = A[aRow * K + aCol];
    } else {
      As[ty][tx] = 0.0f;
    }

    // TODO:
    // Load B tile into Bs.
    // If out of bounds, write 0.
    if (/* TODO */ bRow < K && bCol < N) {
      Bs[ty][tx] = B[bRow * N + bCol];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // TODO:
    // Use shared memory to accumulate partial dot product.
    for (int t = 0; t < TILE; ++t) {
      // TODO:
      // sum += As[ty][t] * Bs[t][tx];
      sum += As[ty][t] * Bs[t][tx];
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

#define TRANSPOSE_TILE = 16

__global__ void transposeTiled(const float *in, float *out, int rows,
                               int cols) {
  // +1 padding helps avoid shared memory bank conflicts.
  __shared__ float tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // TODO:
  // Global coordinates for reading input.
  int x = blockIdx.x * blockDim.x + threadIdx.x; // column in input
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row in input

  // TODO:
  // Load input tile into shared memory.
  if (/* TODO: y < rows && x < cols */y < rows && x < cols) {
    // TODO:
    // tile[ty][tx] = in[y * cols + x];
    tile[ty][tx] = in[y * cols + x];
  }

  __syncthreads();

  // TODO:
  // Compute transposed block coordinates.
  // Swap blockIdx.x and blockIdx.y.
  int transposedX = blockIdx.y * TILE + tx; // TODO
  int transposedY = blockIdx.x * TILE + ty; // TODO

  // TODO:
  // Write transposed tile to output.
  if (/* TODO */) {
    // TODO:
    // out[transposedY * rows + transposedX] = tile[tx][ty];
    out[transposedY * rows + transposedX] = tile[tx][ty];
  }
}

void matrixAddCPU(const std::vector<float> &A, const std::vector<float> &B,
                  std::vector<float> &C, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      int idx = row * cols + col;
      C[idx] = A[idx] + B[idx];
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
  const int rows = 1024;
  const int cols = 1024;
  const int N = rows * cols;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_A(N);
  std::vector<float> h_B(N);
  std::vector<float> h_C(N, 0.0f);
  std::vector<float> h_ref(N, 0.0f);

  // Initialize host matrices.
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      int idx = row * cols + col;
      h_A[idx] = static_cast<float>(row);
      h_B[idx] = static_cast<float>(col);
    }
  }

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  // TODO 2:
  // Allocate device memory for d_A, d_B, d_C.
  //
  CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

  // TODO 3:
  // Copy h_A and h_B from host to device.
  //
  CHECK_CUDA(cudaMemCpy(d_A, h_A.data(), bytes, cudaMemCpyHostToDevice));
  CHECK_CUDA(cudaMemCpy(d_B, h_B.data(), bytes, cudaMemCpyHostToDevice));

  // TODO 4:
  // Choose a 2-D block size.
  //
  // Common choice:
  //   dim3 blockSize(16, 16);
  //
  // Why 16x16?
  //   256 threads per block, a common safe starting point.
  dim3 blockSize(16, 16); // TODO

  // TODO 5:
  // Compute a 2-D grid size.
  //
  // grid.x covers columns.
  // grid.y covers rows.
  //
  dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                (rows + blockSize.y - 1) / blockSize.y); // TODO

  // TODO 6:
  // Launch matrixAddKernel.
  //
  // matrixAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
  matrixAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);

  // TODO 7:
  // Check kernel launch error and synchronize.
  //
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // TODO 8:
  // Copy d_C back to h_C.
  //
  CHECK_CUDA(cudaMemCpy(h_C.data(), d_C, bytes, cudaMemCpyDeviceToHost));

  // CPU reference.
  matrixAddCPU(h_A, h_B, h_ref, rows, cols);

  if (checkResult(h_C, h_ref)) {
    std::cout << "PASS: GPU result matches CPU result." << std::endl;
  } else {
    std::cout << "FAIL: GPU result does not match CPU result." << std::endl;
  }

  // TODO 9:
  // Free device memory.
  //
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));

  return 0;
}