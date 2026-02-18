# GPU-Accelerated Symmetric Matrix Multiplication ($A \times A^T$)

This project explores **High-Performance Computing (HPC)** optimizations for computing the product of a matrix and its transpose ($A \times A^T$). It features a side-by-side comparison of three distinct implementations, scaling from a baseline serial approach to a massively parallel CUDA implementation.



## Project Goals
The primary objective was to minimize execution time by leveraging hardware-specific features:
* **Exploiting Symmetry:** Only the lower triangular part of the result matrix is computed, reducing the total workload by approximately 50%.
* **Memory Locality:** Implementing tiling strategies to maximize cache reuse.
* **Architecture Utilization:** Squeezing maximum GFLOPS out of both multi-core CPUs (SIMD) and NVIDIA GPUs (CUDA).

---

## Tech Stack & Optimizations

### 1. Reference Implementation (CPU)
* **Approach:** Standard triple-nested loop.
* **Purpose:** Serves as the "Gold Standard" for correctness verification.

### 2. Optimized CPU (OpenMP + AVX2)
* **Vectorization:** Uses AVX2 intrinsics and `float8_t` vector types to process 8 floats per instruction.
* **Parallelism:** Multi-threaded execution via **OpenMP** to saturate all available CPU cores.
* **Alignment:** Aligned memory allocation (64-byte) to prevent performance penalties on SIMD loads.

### 3. Optimized GPU (CUDA)
* **Register Tiling:** Implements an $8 \times 8$ micro-kernel to minimize global memory traffic by keeping data in high-speed registers.
* **Memory Coalescing:** Optimized thread-mapping to ensure high-bandwidth memory access.
* **Preprocessing:** Includes a custom transpose and padding kernel to prepare data for optimal GPU throughput.



---

## Performance Benchmarks

The benchmark suite evaluates three distinct hardware scenarios. The GPU demonstrates its true power in high-load "Stress Test" scenarios, achieving over **22x speedup** compared to the optimized CPU.

| Test Case | Dimensions | Purpose | CPU GFLOPS | GPU GFLOPS |
| :--- | :--- | :--- | :--- | :--- |
| **Compute Bound** | $16384 \times 512$ | ALU Throughput | ~254 | ~81 |
| **Memory Bound** | $512 \times 16384$ | RAM Bandwidth | ~177 | ~325 |
| **Stress Test** | $10000 \times 10000$ | Full System Load | ~182 | **~4081** |

### Key Takeaway
While the CPU wins on "thin" matrices due to lower launch overhead, the CUDA implementation scales significantly better once the GPU occupancy is saturated, hitting a peak of **4.1 TFLOPS**.



---

## 🚀 Getting Started

### Prerequisites
* **OS:** Ubuntu 22.04 LTS or newer
* **Compiler:** `g++` (v9+) and `nvcc` (CUDA 11+)
* **Hardware:** NVIDIA GPU (Compute Capability 7.0+)

### Installation & Execution
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/matrix-transpose-multiply.git](https://github.com/yourusername/matrix-transpose-multiply.git)
   cd matrix-transpose-multiply
   
2. **Compile the project:**
   ```bash
   # Compile CPU implementation
   g++ -O3 -fopenmp -mavx2 -c matmul.cc -o cpu.o
   # Compile GPU implementation
   nvcc -O3 -c matmul.cu -o gpu.o
   # Link everything with the main driver
   nvcc -O3 main.cc cpu.o gpu.o -lgomp -o benchmark
   
4. **Run the suite:**
   ```bash
   ./benchmark

