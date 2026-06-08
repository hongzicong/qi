#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "quantize_int8.cuh"

namespace py = pybind11;

static void* to_ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }
template <typename T> static T* typed_ptr(uintptr_t addr) { return reinterpret_cast<T*>(addr); }
static cudaStream_t to_stream(uintptr_t stream) { return reinterpret_cast<cudaStream_t>(stream); }

static void check_cuda_tensor(torch::Tensor const& tensor, char const* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

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

torch::Tensor qi_int8_w8a8_linear(
    torch::Tensor x_bf16_2d,
    torch::Tensor qweight,
    torch::Tensor weight_scales,
    c10::optional<torch::Tensor> bias_opt) {
    check_cuda_tensor(x_bf16_2d, "x_bf16_2d");
    check_cuda_tensor(qweight, "qweight");
    check_cuda_tensor(weight_scales, "weight_scales");
    TORCH_CHECK(x_bf16_2d.scalar_type() == torch::kBFloat16, "x_bf16_2d must be bfloat16.");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt8, "qweight must be int8.");
    TORCH_CHECK(weight_scales.scalar_type() == torch::kFloat32, "weight_scales must be float32.");
    TORCH_CHECK(x_bf16_2d.dim() == 2, "x_bf16_2d must be 2D.");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D.");
    TORCH_CHECK(weight_scales.dim() == 1, "weight_scales must be 1D.");

    auto const M = static_cast<int>(x_bf16_2d.size(0));
    auto const K = static_cast<int>(x_bf16_2d.size(1));
    auto const N = static_cast<int>(qweight.size(0));
    TORCH_CHECK(qweight.size(1) == K, "qweight K dimension must match x_bf16_2d.");
    TORCH_CHECK(weight_scales.size(0) == N, "weight_scales length must match qweight N dimension.");

    auto device = x_bf16_2d.device();
    TORCH_CHECK(qweight.device() == device, "qweight must be on the same device as x_bf16_2d.");
    TORCH_CHECK(weight_scales.device() == device, "weight_scales must be on the same device as x_bf16_2d.");

    auto stream = at::cuda::getCurrentCUDAStream(device.index());
    auto x_int8 = torch::empty({M, K}, torch::TensorOptions().device(device).dtype(torch::kInt8));
    auto act_scales = torch::empty({M}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto out = torch::empty({M, N}, torch::TensorOptions().device(device).dtype(torch::kBFloat16));

    qi_quantize_int8_rowwise(
        reinterpret_cast<__nv_bfloat16*>(x_bf16_2d.data_ptr<at::BFloat16>()),
        x_int8.data_ptr<int8_t>(),
        act_scales.data_ptr<float>(),
        M,
        K,
        stream.stream());

    int rc = 0;
    if (bias_opt.has_value() && bias_opt->defined()) {
        auto bias = *bias_opt;
        check_cuda_tensor(bias, "bias");
        TORCH_CHECK(bias.device() == device, "bias must be on the same device as x_bf16_2d.");
        TORCH_CHECK(bias.scalar_type() == torch::kBFloat16, "bias must be bfloat16.");
        TORCH_CHECK(bias.numel() == N, "bias length must match qweight N dimension.");
        rc = qi_cutlass_int8_rowwise_bf16out_bias(
            x_int8.data_ptr<int8_t>(),
            qweight.data_ptr<int8_t>(),
            act_scales.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            bias.data_ptr<at::BFloat16>(),
            out.data_ptr<at::BFloat16>(),
            M,
            N,
            K,
            stream.stream());
    } else {
        rc = qi_cutlass_int8_rowwise_bf16out(
            x_int8.data_ptr<int8_t>(),
            qweight.data_ptr<int8_t>(),
            act_scales.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            out.data_ptr<at::BFloat16>(),
            M,
            N,
            K,
            stream.stream());
    }
    TORCH_CHECK(rc == 0, "INT8 W8A8 CUTLASS GEMM failed with status ", rc, ".");
    return out;
}

TORCH_LIBRARY(qi_int8_w8a8, m) {
    m.def("linear(Tensor x_bf16_2d, Tensor qweight, Tensor weight_scales, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(qi_int8_w8a8, CUDA, m) {
    m.impl("linear", &qi_int8_w8a8_linear);
}

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
