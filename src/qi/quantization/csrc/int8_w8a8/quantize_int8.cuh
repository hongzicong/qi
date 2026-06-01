#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

void qi_quantize_int8_rowwise(
    const __nv_bfloat16* input,
    int8_t* output,
    float* scales,
    int rows,
    int cols,
    cudaStream_t stream = 0);
