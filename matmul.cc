#include <cmath>
#include <vector>
#include <omp.h>
#include <memory>
#include <emmintrin.h>
#include <chrono>
#include <iostream>
#include "timer.h"

typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));

/*
- Input data matrix A has dimensions ny, nx
- Output A * A^T has dimensions ny, ny
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled due to symmetry
*/

void cpu_matmul(int ny, int nx, const float *data, float *result) {
    // Timing the entire CPU computation
    WallTimer t("CPU_Total");

    // vectors per input row
    int na = (nx + 8 - 1) / 8;
    // block size
    constexpr int nd = 7;
    // how many blocks of rows
    int nc = (ny + nd - 1) / nd;
    // number of rows after padding
    int ncd = nc * nd;

    // The input data, padded, converted to vectors (represents A)
    auto temp = static_cast<float8_t*>(_mm_malloc(ncd * na * sizeof(float8_t), 64));
    constexpr float8_t zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    {
        // Prepare the data: transpose, convert to vectors, and pad columns
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < ny; ++y) { 
            for (int ka = 0; ka < na; ++ka) {
                float8_t vec = zero;
                for (int kb = 0; kb < 8; ++kb) {
                    int x = kb + ka * 8;
                    if (x < nx) {
                        vec[kb] = data[x + y * nx];
                    }
                }
                temp[ka + y * na] = vec;
            }
        }
    }

    {
        // Padding
        #pragma omp parallel for
        for (int j = ny; j < ncd; ++j) {
            for (int ka = 0; ka < na; ++ka) {
                temp[ka + j * na] = zero;
            }
        }
    }
    {
        // Matrix Multiplication: Compute A * A^T using the prepared data in temp, store results in result
        #pragma omp parallel for schedule(dynamic, 1)
        for (int ic = 0; ic < nc; ++ic) { // ic:th row block, rows ic * 5 to ic * 5 + 4
            const int i_start = ic * nd;
            const int i_end = std::min(i_start + nd, ny);
            
            for (int jc = 0; jc <= ic; ++jc) { // jc:th row block, rows jc * 5 to jc * 5 + 4
                const int j_start = jc * nd;
                const int j_end = std::min(j_start + nd, ny);
                
                float8_t vv00 = {0.0f}, vv01 = {0.0f}, vv02 = {0.0f}, vv03 = {0.0f}, vv04 = {0.0f}, vv05 = {0.0f}, vv06 = {0.0f};//, vv07 = {0.0}, vv08 = {0.0}, vv09 = {0.0};
                float8_t vv10 = {0.0f}, vv11 = {0.0f}, vv12 = {0.0f}, vv13 = {0.0f}, vv14 = {0.0f}, vv15 = {0.0f}, vv16 = {0.0f};//, vv17 = {0.0}, vv18 = {0.0}, vv19 = {0.0};
                float8_t vv20 = {0.0f}, vv21 = {0.0f}, vv22 = {0.0f}, vv23 = {0.0f}, vv24 = {0.0f}, vv25 = {0.0f}, vv26 = {0.0f};//, vv27 = {0.0}, vv28 = {0.0}, vv29 = {0.0};
                float8_t vv30 = {0.0f}, vv31 = {0.0f}, vv32 = {0.0f}, vv33 = {0.0f}, vv34 = {0.0f}, vv35 = {0.0f}, vv36 = {0.0f};//, vv37 = {0.0}, vv38 = {0.0}, vv39 = {0.0};
                float8_t vv40 = {0.0f}, vv41 = {0.0f}, vv42 = {0.0f}, vv43 = {0.0f}, vv44 = {0.0f}, vv45 = {0.0f}, vv46 = {0.0f};//, vv47 = {0.0}, vv48 = {0.0}, vv49 = {0.0};
                float8_t vv50 = {0.0f}, vv51 = {0.0f}, vv52 = {0.0f}, vv53 = {0.0f}, vv54 = {0.0f}, vv55 = {0.0f}, vv56 = {0.0f};//, vv57 = {0.0}, vv58 = {0.0}, vv59 = {0.0};
                float8_t vv60 = {0.0f}, vv61 = {0.0f}, vv62 = {0.0f}, vv63 = {0.0f}, vv64 = {0.0f}, vv65 = {0.0f}, vv66 = {0.0f};//, vv67 = {0.0}, vv68 = {0.0}, vv69 = {0.0};
                //float8_t vv70 = {0.0}, vv71 = {0.0}, vv72 = {0.0}, vv73 = {0.0}, vv74 = {0.0}, vv75 = {0.0}, vv76 = {0.0};//, vv77 = {0.0}, vv78 = {0.0}, vv79 = {0.0};
                //float8_t vv80 = {0.0}, vv81 = {0.0}, vv82 = {0.0}, vv83 = {0.0}, vv84 = {0.0}, vv85 = {0.0}, vv86 = {0.0};//, vv87 = {0.0}, vv88 = {0.0}, vv89 = {0.0};
                //float8_t vv90 = {0.0}, vv91 = {0.0}, vv92 = {0.0}, vv93 = {0.0}, vv94 = {0.0}, vv95 = {0.0}, vv96 = {0.0};//, vv97 = {0.0}, vv98 = {0.0}, vv99 = {0.0};
                // 7 * 7 = 49 float8_t vectors, remaining spots in register = 64-7*7 = 15
                
                for (int ka = 0; ka < na; ++ka) {
                    constexpr int PF = 16;
                    __builtin_prefetch(&temp[ka + PF + i_start * na]);
                    __builtin_prefetch(&temp[ka + PF + j_start * na]);

                    const float8_t a0 = temp[ka + i_start * na];
                    const float8_t a1 = temp[ka + (i_start + 1) * na];
                    const float8_t a2 = temp[ka + (i_start + 2) * na];
                    const float8_t a3 = temp[ka + (i_start + 3) * na];
                    const float8_t a4 = temp[ka + (i_start + 4) * na];
                    const float8_t a5 = temp[ka + (i_start + 5) * na];
                    const float8_t a6 = temp[ka + (i_start + 6) * na];
                    const float8_t b0 = temp[ka + j_start * na];
                    const float8_t b1 = temp[ka + (j_start + 1) * na];
                    const float8_t b2 = temp[ka + (j_start + 2) * na];
                    const float8_t b3 = temp[ka + (j_start + 3) * na];
                    const float8_t b4 = temp[ka + (j_start + 4) * na];
                    const float8_t b5 = temp[ka + (j_start + 5) * na];
                    const float8_t b6 = temp[ka + (j_start + 6) * na];
                    
                    vv00 += a0 * b0; vv01 += a0 * b1; vv02 += a0 * b2; vv03 += a0 * b3; vv04 += a0 * b4; vv05 += a0 * b5; vv06 += a0 * b6;
                    vv10 += a1 * b0; vv11 += a1 * b1; vv12 += a1 * b2; vv13 += a1 * b3; vv14 += a1 * b4; vv15 += a1 * b5; vv16 += a1 * b6;
                    vv20 += a2 * b0; vv21 += a2 * b1; vv22 += a2 * b2; vv23 += a2 * b3; vv24 += a2 * b4; vv25 += a2 * b5; vv26 += a2 * b6;
                    vv30 += a3 * b0; vv31 += a3 * b1; vv32 += a3 * b2; vv33 += a3 * b3; vv34 += a3 * b4; vv35 += a3 * b5; vv36 += a3 * b6;
                    vv40 += a4 * b0; vv41 += a4 * b1; vv42 += a4 * b2; vv43 += a4 * b3; vv44 += a4 * b4; vv45 += a4 * b5; vv46 += a4 * b6;
                    vv50 += a5 * b0; vv51 += a5 * b1; vv52 += a5 * b2; vv53 += a5 * b3; vv54 += a5 * b4; vv55 += a5 * b5; vv56 += a5 * b6;
                    vv60 += a6 * b0; vv61 += a6 * b1; vv62 += a6 * b2; vv63 += a6 * b3; vv64 += a6 * b4; vv65 += a6 * b5; vv66 += a6 * b6;
                }

                const float8_t vv[7 * 7] = {vv00, vv01, vv02, vv03, vv04, vv05, vv06, // row ic * 5 with rows jc * 5 to jc * 5 + 6,      j = 0, i in {0, ..., 6}
                                            vv10, vv11, vv12, vv13, vv14, vv15, vv16, // row ic * 5 + 1 with rows jc * 5 to jc * 5 + 6,  j = 1, i in {0, ..., 6}
                                            vv20, vv21, vv22, vv23, vv24, vv25, vv26, // row ic * 5 + 2 with rows jc * 5 to jc * 5 + 6,  j = 2, i in {0, ..., 6}
                                            vv30, vv31, vv32, vv33, vv34, vv35, vv36, // row ic * 5 + 3 with rows jc * 5 to jc * 5 + 6,  j = 3, i in {0, ..., 6}
                                            vv40, vv41, vv42, vv43, vv44, vv45, vv46, // row ic * 5 + 4 with rows jc * 5 to jc * 5 + 6,  j = 4, i in {0, ..., 6}
                                            vv50, vv51, vv52, vv53, vv54, vv55, vv56, // row ic * 5 + 5 with rows jc * 5 to jc * 5 + 6,  j = 5, i in {0, ..., 6}
                                            vv60, vv61, vv62, vv63, vv64, vv65, vv66  // row ic * 5 + 6 with rows jc * 5 to jc * 5 + 6,  j = 6, i in {0, ..., 6}
                                    };
                
                // Store results with correct indexing
                for (int i = 0; i < i_end - i_start; ++i) {
                    for (int j = 0; j < j_end - j_start; ++j) {
                        const int row = i_start + i;
                        const int col = j_start + j;
                        if (col <= row) {
                            float dot_product = vv[j + i * nd][0] + vv[j + i * nd][1] + vv[j + i * nd][2] + vv[j + i * nd][3] + vv[j + i * nd][4] + vv[j + i * nd][5] + vv[j + i * nd][6] + vv[j + i * nd][7];
                            result[row + col * ny] = dot_product;
                        }
                    }
                }
            }
        }
        _mm_free(temp);
    }
}