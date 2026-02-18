#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <mm_malloc.h>

extern void cpu_matmul(int ny, int nx, const float *data, float *result);
extern void gpu_matmul(int ny, int nx, const float *data, float *result);

struct TestCase {
    std::string name;
    int ny;
    int nx;
};

int main() {
    // Expanded test suite to see scaling
    std::vector<TestCase> tests = {
        // Small Scale
        {"512x512",     512,   512},
        {"1024x1024",   1024,  1024},
        {"1536x1536",   1536,  1536},
        {"2048x2048",   2048,  2048},

        // Mid Scale
        {"3072x3072",   3072,  3072},
        {"4096x4096",   4096,  4096},
        {"5120x5120",   5120,  5120},
        {"6144x6144",   6144,  6144},
        {"7168x7168",   7168,  7168},
        {"8192x8192",   8192,  8192},

        // Large Scale
        {"9216x9216",   9216,  9216},
        {"10000x10000", 10000, 10000},
        {"11264x11264", 11264, 11264},
        {"12288x12288", 12288, 12288},
        {"14336x14336", 14336, 14336},
        {"16384x16384", 16384, 16384}, // ~1GB of Input Data

        // Rectangular / Bottleneck Tests
        {"Compute Bound (Thin)", 16384, 512},   
        {"Memory Bound (Wide)",  512,   16384},  
        {"Extreme Ratio",        32768, 64},   // High launch overhead test
        {"Long Vector",          64,    32768}    // Streaming bandwidth test
    };

    // CSV Header for easy plotting
    std::cout << "Name,MB,CPU_ms,GPU_ms" << std::endl;

    for (const auto& test : tests) {
        float mb = (float)test.ny * test.nx * sizeof(float) / (1024.0f * 1024.0f);

        float* data = (float*)_mm_malloc(test.ny * test.nx * sizeof(float), 64);
        float* cpu_res = (float*)_mm_malloc(test.ny * test.ny * sizeof(float), 64);
        float* gpu_res = (float*)_mm_malloc(test.ny * test.ny * sizeof(float), 64);

        // Initialize with random data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < test.ny * test.nx; ++i) data[i] = dis(gen);

        // CPU Time
        auto s1 = std::chrono::high_resolution_clock::now();
        cpu_matmul(test.ny, test.nx, data, cpu_res);
        auto e1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

        // GPU Time
        auto s2 = std::chrono::high_resolution_clock::now();
        gpu_matmul(test.ny, test.nx, data, gpu_res);
        auto e2 = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

        // Print CSV row
        std::cout << test.name << "," << mb << "," << cpu_ms << "," << gpu_ms << std::endl;

        _mm_free(data); _mm_free(cpu_res); _mm_free(gpu_res);
    }
    return 0;
}