#include <vector>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"


static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

/*
- Input data matrix A has dimensions ny, nx
- Output A * A^T has dimensions ny, ny
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- result[i + j*ny] stores the dot product of row i and row j
- only parts with 0 <= j <= i < ny need to be filled due to symmetry
*/

// Preprocessing kernel: Transposes and pads the matrix for optimized access in mykernel
__global__ void myppkernel(const float* r, float* d, int nx, int ny, int nn, int ny_padded) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col

    if (i < ny && j < nx) {
        d[i + j * ny_padded] = r[j + i * nx];
    }
}

// Matrix multiplication kernel (A * A^T)
__global__ void mykernel(float* r, const float* d, int nx, int ny, int nn, int ny_padded) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    if (jc > ic) return;

    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = 0.0f;
        }
    }

    for (int k = 0; k < nn; ++k) {
        float x[8], y[8];

        for (int ib = 0; ib < 8; ++ib) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = d[ny_padded*k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = d[ny_padded*k + j];
        }

        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                v[ib][jb] += x[ib] * y[jb];
            }
        }
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < ny && j <= i) {
                r[i + j * ny] = v[ib][jb];
            }
        }
    }
}

void gpu_matmul(int ny, int nx, const float *data, float *result) {
    // Timing the entire GPU computation
    WallTimer t("GPU_Total");
    
    if (nx <= 0 || ny <= 0) return;

    // 1. Allocate and Copy Input Data
    float* dGPU_data = NULL;
    CHECK(cudaMalloc((void**)&dGPU_data, nx * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU_data, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    int nn = roundup(nx, 64);
    int ny_padded = roundup(ny, 64);

    // 2. Allocate memory for padded/transposed data and result
    float* dGPU_transposed = NULL;
    CHECK(cudaMalloc((void**)&dGPU_transposed, nn * ny_padded * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    
    CHECK(cudaMemset(dGPU_transposed, 0, nn * ny_padded * sizeof(float)));
    CHECK(cudaMemset(rGPU, 0, ny * ny * sizeof(float)));

    // 3. Transpose directly from raw dGPU_data
    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid(divup(nx, 32), divup(ny, 32));
        myppkernel<<<dimGrid, dimBlock>>>(dGPU_data, dGPU_transposed, nx, ny, nn, ny_padded);
        CHECK(cudaGetLastError());
    }

    // 4. Compute A * A^T
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(divup(ny_padded, 64), divup(ny_padded, 64));
        mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU_transposed, nx, ny, nn, ny_padded);
        CHECK(cudaGetLastError());
    }

    // 5. Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(dGPU_transposed));
    CHECK(cudaFree(rGPU));
    CHECK(cudaFree(dGPU_data));
    cudaDeviceSynchronize();
}