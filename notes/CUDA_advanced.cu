#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " : " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;

// For signed int ascending sort:
//
// Normal unsigned radix sort would put all positive numbers before negative
// numbers because of two's-complement representation.
//
// Example:
//   -1 as unsigned is very large.
//
// So we flip the sign bit:
//
//   signed int order:
//       negative values, then non-negative values
//
//   transformed unsigned order after xor 0x80000000:
//       negative values map to [0, 2^31 - 1]
//       non-negative values map to [2^31, 2^32 - 1]
//
// Therefore sorting by transformedKey(x) gives correct signed-int order.
__device__ __forceinline__ unsigned int transformedKey(int x) {
    return static_cast<unsigned int>(x) ^ 0x80000000u;
}

// Simple block-wide inclusive scan using shared memory.
//
// Input:
//   x: each thread's value, usually 0 or 1
//
// Output:
//   exclusive prefix sum for this thread
//
// After the function returns:
//   s[threadIdx.x] contains the inclusive scan value.
//   s[blockDim.x - 1] contains the total sum for the block.
//
// This is Hillis-Steele style scan.
// Not the fastest possible scan, but very readable for interviews.
__device__ unsigned int blockExclusiveScan(unsigned int x, unsigned int* s) {
    int tid = threadIdx.x;

    s[tid] = x;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        unsigned int add = 0;

        if (tid >= offset) {
            add = s[tid - offset];
        }

        __syncthreads();

        s[tid] += add;

        __syncthreads();
    }

    // s[tid] is inclusive scan.
    // exclusive scan = inclusive - own value.
    return s[tid] - x;
}

// Generic exclusive scan for an array of unsigned int.
//
// Each block scans BLOCK_SIZE elements.
// blockSums[blockIdx.x] receives the total sum of that block.
//
// This kernel alone only produces correct exclusive scans inside each block.
// The host-side recursive function below scans blockSums and adds the scanned
// block offsets back.
__global__ void exclusiveScanBlockKernel(const unsigned int* in,
                                         unsigned int* out,
                                         unsigned int* blockSums,
                                         int n) {
    extern __shared__ unsigned int s[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    unsigned int x = 0;
    if (idx < n) {
        x = in[idx];
    }

    unsigned int exclusive = blockExclusiveScan(x, s);

    if (idx < n) {
        out[idx] = exclusive;
    }

    if (tid == 0 && blockSums != nullptr) {
        blockSums[blockIdx.x] = s[blockDim.x - 1];
    }
}

// Add scanned block offsets back to each element.
//
// After exclusiveScanBlockKernel, each block has a local exclusive scan.
// This kernel converts local scan into global scan:
//
//   out[i] += scannedBlockSums[blockIdx.x]
//
__global__ void addBlockOffsetsKernel(unsigned int* out,
                                      const unsigned int* scannedBlockSums,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] += scannedBlockSums[blockIdx.x];
    }
}

// Recursive device-wide exclusive scan.
//
// This is not as optimized as CUB, but it is complete and educational.
//
// Example:
//
//   input:  [3, 1, 4, 2]
//   output: [0, 3, 4, 8]
//
void exclusiveScanDevice(const unsigned int* d_in,
                         unsigned int* d_out,
                         int n) {
    if (n <= 0) return;

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (numBlocks == 1) {
        exclusiveScanBlockKernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(
            d_in, d_out, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    unsigned int* d_blockSums = nullptr;
    unsigned int* d_scannedBlockSums = nullptr;

    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_scannedBlockSums, numBlocks * sizeof(unsigned int)));

    exclusiveScanBlockKernel<<<numBlocks, BLOCK_SIZE,
                               BLOCK_SIZE * sizeof(unsigned int)>>>(
        d_in, d_out, d_blockSums, n);
    CUDA_CHECK(cudaGetLastError());

    // Recursively scan the per-block sums.
    exclusiveScanDevice(d_blockSums, d_scannedBlockSums, numBlocks);

    // Add scanned block offsets to each block's local result.
    addBlockOffsetsKernel<<<numBlocks, BLOCK_SIZE>>>(
        d_out, d_scannedBlockSums, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_scannedBlockSums));
}

// For the current radix bit, count how many zeros are in each block.
//
// For one radix pass, we stable-partition by one bit:
//
//   bit = 0 elements go to the front
//   bit = 1 elements go after all zeros
//
// This kernel only computes:
//
//   blockZeroCounts[blockIdx.x] = number of zero-bit elements in this block
//
__global__ void countZerosPerBlockKernel(const int* in,
                                         unsigned int* blockZeroCounts,
                                         int n,
                                         int bit) {
    extern __shared__ unsigned int s[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    unsigned int zeroFlag = 0;

    if (idx < n) {
        unsigned int key = transformedKey(in[idx]);
        unsigned int b = (key >> bit) & 1u;
        zeroFlag = (b == 0u) ? 1u : 0u;
    }

    blockExclusiveScan(zeroFlag, s);

    if (tid == 0) {
        blockZeroCounts[blockIdx.x] = s[blockDim.x - 1];
    }
}

// Stable scatter for one radix bit.
//
// Given:
//   zeroOffsets[block] = number of zeros in all previous blocks
//   totalZeros         = number of zeros in the entire array
//
// For each element:
//
// If bit == 0:
//   position = zeros before this block
//            + zeros before this element inside this block
//
// If bit == 1:
//   position = totalZeros
//            + ones before this block
//            + ones before this element inside this block
//
// The important thing:
//   This is stable.
//   Elements with the same bit keep their original relative order.
//
__global__ void scatterByBitKernel(const int* in,
                                   int* out,
                                   const unsigned int* zeroOffsets,
                                   int n,
                                   int bit,
                                   unsigned int totalZeros) {
    extern __shared__ unsigned int s[];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int idx = blockStart + tid;

    unsigned int zeroFlag = 0;
    unsigned int b = 0;

    if (idx < n) {
        unsigned int key = transformedKey(in[idx]);
        b = (key >> bit) & 1u;
        zeroFlag = (b == 0u) ? 1u : 0u;
    }

    unsigned int zerosBeforeInBlock = blockExclusiveScan(zeroFlag, s);

    if (idx >= n) {
        return;
    }

    unsigned int zerosBeforeThisBlock = zeroOffsets[blockIdx.x];

    if (b == 0u) {
        unsigned int pos = zerosBeforeThisBlock + zerosBeforeInBlock;
        out[pos] = in[idx];
    } else {
        // Number of valid previous elements inside the block is tid.
        // Among those, zerosBeforeInBlock are zeros.
        // Therefore the number of previous ones inside the block is:
        unsigned int onesBeforeInBlock = tid - zerosBeforeInBlock;

        // All previous blocks are full except possibly the current last block.
        // Since this is about previous blocks only, blockStart is valid.
        unsigned int onesBeforeThisBlock =
            static_cast<unsigned int>(blockStart) - zerosBeforeThisBlock;

        unsigned int pos =
            totalZeros + onesBeforeThisBlock + onesBeforeInBlock;

        out[pos] = in[idx];
    }
}

// Public callable function.
//
// Input:
//   d_input  : device pointer to input int array
//   d_output : device pointer to output int array
//   n        : number of elements
//
// Behavior:
//   Sorts d_input ascending and writes result to d_output.
//   d_input is not modified.
//   d_input and d_output may be the same pointer.
//
void gpuRadixSortInt(const int* d_input, int* d_output, int n) {
    if (n <= 0) return;

    if (n == 1) {
        CUDA_CHECK(cudaMemcpy(d_output, d_input, sizeof(int),
                              cudaMemcpyDeviceToDevice));
        return;
    }

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* d_bufA = nullptr;
    int* d_bufB = nullptr;

    unsigned int* d_blockZeroCounts = nullptr;
    unsigned int* d_zeroOffsets = nullptr;

    CUDA_CHECK(cudaMalloc(&d_bufA, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bufB, n * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_blockZeroCounts,
                          numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_zeroOffsets,
                          numBlocks * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_bufA, d_input, n * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    int* src = d_bufA;
    int* dst = d_bufB;

    // 32 passes for 32-bit int.
    //
    // This is least-significant-bit radix sort.
    // Each pass is a stable partition by one bit.
    for (int bit = 0; bit < 32; ++bit) {
        countZerosPerBlockKernel<<<numBlocks, BLOCK_SIZE,
                                   BLOCK_SIZE * sizeof(unsigned int)>>>(
            src, d_blockZeroCounts, n, bit);
        CUDA_CHECK(cudaGetLastError());

        // zeroOffsets[block] = number of zeros in all previous blocks.
        exclusiveScanDevice(d_blockZeroCounts, d_zeroOffsets, numBlocks);

        // totalZeros = zeroOffsets[lastBlock] + zeroCounts[lastBlock]
        unsigned int lastOffset = 0;
        unsigned int lastCount = 0;

        CUDA_CHECK(cudaMemcpy(&lastOffset,
                              d_zeroOffsets + (numBlocks - 1),
                              sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(&lastCount,
                              d_blockZeroCounts + (numBlocks - 1),
                              sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        unsigned int totalZeros = lastOffset + lastCount;

        scatterByBitKernel<<<numBlocks, BLOCK_SIZE,
                             BLOCK_SIZE * sizeof(unsigned int)>>>(
            src, dst, d_zeroOffsets, n, bit, totalZeros);
        CUDA_CHECK(cudaGetLastError());

        std::swap(src, dst);
    }

    CUDA_CHECK(cudaMemcpy(d_output, src, n * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_bufA));
    CUDA_CHECK(cudaFree(d_bufB));
    CUDA_CHECK(cudaFree(d_blockZeroCounts));
    CUDA_CHECK(cudaFree(d_zeroOffsets));
}

int main() {
    constexpr int N = 1 << 20;

    std::vector<int> h_input(N);
    std::vector<int> h_output(N);
    std::vector<int> h_reference(N);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    for (int i = 0; i < N; ++i) {
        h_input[i] = dist(rng);
    }

    h_reference = h_input;
    std::sort(h_reference.begin(), h_reference.end());

    int* d_input = nullptr;
    int* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(int),
                          cudaMemcpyHostToDevice));

    gpuRadixSortInt(d_input, d_output, N);

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool ok = (h_output == h_reference);

    if (ok) {
        std::cout << "Radix sort correct!" << std::endl;
    } else {
        std::cout << "Radix sort WRONG!" << std::endl;

        for (int i = 0; i < 20; ++i) {
            std::cout << "i = " << i
                      << ", gpu = " << h_output[i]
                      << ", cpu = " << h_reference[i]
                      << std::endl;
        }
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return ok ? 0 : 1;
}