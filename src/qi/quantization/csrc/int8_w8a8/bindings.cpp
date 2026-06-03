#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "quantize_int8.cuh"

namespace py = pybind11;

static void* to_ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }
template <typename T> static T* typed_ptr(uintptr_t addr) { return reinterpret_cast<T*>(addr); }
static cudaStream_t to_stream(uintptr_t stream) { return reinterpret_cast<cudaStream_t>(stream); }

extern "C" int qi_cutlass_int8_rowwise_bf16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

extern "C" int qi_cutlass_int8_rowwise_bf16out_t64x128(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

extern "C" int qi_cutlass_int8_rowwise_bf16out_bias(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void const* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

extern "C" int qi_cutlass_int8_rowwise_bf16out_bias_t64x128(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void const* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

extern "C" int qi_cutlass_int8_rowwise_bf16out_t64x256(
    void const* A, void const* B, void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream);
extern "C" int qi_cutlass_int8_rowwise_bf16out_bias_t64x256(
    void const* A, void const* B, void const* act_scale, void const* weight_scale,
    void const* bias, void* D, int M, int N, int K, cudaStream_t stream);

extern "C" int qi_cutlass_int8_rowwise_bf16out_t32x128(
    void const* A, void const* B, void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream);
extern "C" int qi_cutlass_int8_rowwise_bf16out_bias_t32x128(
    void const* A, void const* B, void const* act_scale, void const* weight_scale,
    void const* bias, void* D, int M, int N, int K, cudaStream_t stream);

extern "C" int qi_cutlass_int8_rowwise_bf16out_t128x64(
    void const* A, void const* B, void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream);
extern "C" int qi_cutlass_int8_rowwise_bf16out_bias_t128x64(
    void const* A, void const* B, void const* act_scale, void const* weight_scale,
    void const* bias, void* D, int M, int N, int K, cudaStream_t stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "quantize_int8_rowwise",
        [](uintptr_t input, uintptr_t output, uintptr_t scales, int rows, int cols, uintptr_t stream) {
            qi_quantize_int8_rowwise(
                typed_ptr<__nv_bfloat16>(input),
                typed_ptr<int8_t>(output),
                typed_ptr<float>(scales),
                rows,
                cols,
                to_stream(stream));
        },
        py::arg("input"),
        py::arg("output"),
        py::arg("scales"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("act_scale"),
        py::arg("weight_scale"),
        py::arg("D"),
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_t64x128",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_t64x128(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("act_scale"),
        py::arg("weight_scale"),
        py::arg("D"),
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_bias",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t bias, uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_bias(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(bias), to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("act_scale"),
        py::arg("weight_scale"),
        py::arg("bias"),
        py::arg("D"),
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_bias_t64x128",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t bias, uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_bias_t64x128(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(bias), to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"),
        py::arg("B"),
        py::arg("act_scale"),
        py::arg("weight_scale"),
        py::arg("bias"),
        py::arg("D"),
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_t64x256",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_t64x256(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("act_scale"), py::arg("weight_scale"),
        py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_bias_t64x256",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t bias, uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_bias_t64x256(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(bias), to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("act_scale"), py::arg("weight_scale"),
        py::arg("bias"), py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_t32x128",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_t32x128(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("act_scale"), py::arg("weight_scale"),
        py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_bias_t32x128",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t bias, uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_bias_t32x128(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(bias), to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("act_scale"), py::arg("weight_scale"),
        py::arg("bias"), py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_t128x64",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_t128x64(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("act_scale"), py::arg("weight_scale"),
        py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def(
        "cutlass_int8_rowwise_bf16out_bias_t128x64",
        [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
           uintptr_t bias, uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return qi_cutlass_int8_rowwise_bf16out_bias_t128x64(
                to_ptr(A), to_ptr(B), to_ptr(act_scale), to_ptr(weight_scale),
                to_ptr(bias), to_ptr(D), M, N, K, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("act_scale"), py::arg("weight_scale"),
        py::arg("bias"), py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);
}
