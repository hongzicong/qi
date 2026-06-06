#include "quantize_int8.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void qi_quantize_int8_rowwise_kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* in_row = input + static_cast<size_t>(row) * cols;
    int8_t* out_row = output + static_cast<size_t>(row) * cols;

    float tmax = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        tmax = fmaxf(tmax, fabsf(__bfloat162float(in_row[j])));
    }

    for (int off = 16; off > 0; off >>= 1) {
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
    }

    __shared__ float warp_max[8];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    if (lid == 0) {
        warp_max[wid] = tmax;
    }
    __syncthreads();

    if (wid == 0) {
        tmax = (lid < (blockDim.x >> 5)) ? warp_max[lid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
        }
    }

    __shared__ float scale_s;
    if (threadIdx.x == 0) {
        float s = fmaxf(tmax / 127.0f, 1e-10f);
        scales[row] = s;
        scale_s = s;
    }
    __syncthreads();

    float inv_s = 1.0f / scale_s;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = __bfloat162float(in_row[j]) * inv_s;
        int q = __float2int_rn(v);
        q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
        out_row[j] = static_cast<int8_t>(q);
    }
}

void qi_quantize_int8_rowwise(
    const __nv_bfloat16* input,
    int8_t* output,
    float* scales,
    int rows,
    int cols,
    cudaStream_t stream)
{
    int threads = (cols < 256) ? cols : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    qi_quantize_int8_rowwise_kernel<<<rows, threads, 0, stream>>>(
        input, output, scales, rows, cols);
}
