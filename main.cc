#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <mm_malloc.h>
#include "timer.h"

extern void cpu_matmul(int ny, int nx, const float *data, float *result);
extern void gpu_matmul(int ny, int nx, const float *data, float *result);

struct TestCase {
    std::string name;
    int ny;
    int nx;
    std::string purpose;
};

void reference_matmul(int ny, int nx, const float* data, float* result) {
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < nx; ++k) sum += data[k + i * nx] * data[k + j * nx];
            result[i + j * ny] = sum;
        }
    }
}

double calculate_gflops(int ny, int nx, double ms) {
    if (ms <= 0) return 0;
    double ops = (double)ny * (ny + 1) / 2.0 * nx * 2.0; 
    return ops / (ms * 1.0e6);
}

int main() {
    std::vector<TestCase> tests = {
        {"Compute Bound", 16384, 512,   "Tests ALU/math throughput"},
        {"Memory Bound",  512,   16384, "Tests RAM bandwidth"},
        {"Stress Test",   10000, 10000, "Real-world high load"}
    };

    std::cout << std::fixed << std::setprecision(2);

    for (const auto& test : tests) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "TEST: " << test.name << " (" << test.ny << "x" << test.nx << ")\n";
        std::cout << "Purpose: " << test.purpose << "\n";
        std::cout << std::string(60, '-') << "\n";

        // Allocate
        float* data = (float*)_mm_malloc(test.ny * test.nx * sizeof(float), 64);
        float* ref_res = (float*)_mm_malloc(test.ny * test.ny * sizeof(float), 64);
        float* cpu_res = (float*)_mm_malloc(test.ny * test.ny * sizeof(float), 64);
        float* gpu_res = (float*)_mm_malloc(test.ny * test.ny * sizeof(float), 64);

        // Initialize
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < test.ny * test.nx; ++i) data[i] = dis(gen);

        // --- Execution & Timing ---
        auto run = [&](const std::string& label, auto func, float* res) {
            auto start = std::chrono::high_resolution_clock::now();
            func(test.ny, test.nx, data, res);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << std::left << std::setw(15) << label << ": " 
                      << std::right << std::setw(10) << ms << " ms (" 
                      << std::setw(8) << calculate_gflops(test.ny, test.nx, ms) << " GFLOPS)\n";
        };

        // Skip reference for Stress Test to save time
        if (test.ny <= 4096) run("Reference", reference_matmul, ref_res);
        else std::cout << "Reference      : Skipped (Too slow for large Ny)\n";

        run("Optimized CPU", cpu_matmul, cpu_res);
        run("Optimized GPU", gpu_matmul, gpu_res);

        _mm_free(data); _mm_free(ref_res); _mm_free(cpu_res); _mm_free(gpu_res);
    }

    return 0;
}