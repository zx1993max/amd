from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    import torch
    import os

    if not hasattr(custom_kernel, '_module'):
        from torch.utils.cpp_extension import load_inline

        hip_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Software FP4 quantization - exact match of aiter Triton _mxfp4_quant_op
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

__global__ void mxfp4_quant_kernel(
    const unsigned short* __restrict__ A,     // [M, K] bf16 as uint16
    unsigned char*        __restrict__ A_fp4,  // [M, K/2] packed fp4x2
    unsigned char*        __restrict__ A_scale, // flat shuffled
    int M, int K, int scaleN_pad
) {
    int m = blockIdx.x;
    int qb = blockIdx.y * blockDim.x + threadIdx.x;
    int num_qblocks = K / 32;

    if (m >= M || qb >= num_qblocks) return;

    int k_base = qb * 32;
    const unsigned short* row = A + m * K + k_base;

    // Load 32 bf16 -> float, compute amax
    float vals[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        vals[i] = bf16_to_f32(row[i]);
        float av = fabsf(vals[i]);
        if (av > amax) amax = av;
    }

    // Bankers rounding of amax (same as Triton kernel)
    unsigned int ai = __float_as_uint(amax);
    ai = (ai + 0x200000u) & 0xFF800000u;
    amax = __uint_as_float(ai);

    // scale_e8m0_unbiased = floor(log2(amax)) - 2, clamped to [-127, 127]
    float log2_amax = log2f(amax);
    float scale_unbiased = floorf(log2_amax) - 2.0f;
    if (scale_unbiased < -127.0f) scale_unbiased = -127.0f;
    if (scale_unbiased > 127.0f) scale_unbiased = 127.0f;

    // E8M0 byte
    int e8m0 = (int)scale_unbiased + 127;
    if (e8m0 < 0) e8m0 = 0;
    if (e8m0 > 254) e8m0 = 254;

    // quant_scale = 2^(-scale_unbiased)
    float quant_scale = exp2f(-scale_unbiased);

    // Quantize each value to FP4 E2M1
    unsigned char fp4_vals[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float qx = vals[i] * quant_scale;
        unsigned int qx_bits = __float_as_uint(qx);

        // Extract sign
        unsigned int sign = qx_bits & 0x80000000u;
        qx_bits ^= sign;  // make positive

        float qx_pos = __uint_as_float(qx_bits);
        unsigned char e2m1;

        if (qx_pos >= 6.0f) {
            // Saturate
            e2m1 = 0x7;
        } else if (qx_pos < 1.0f) {
            // Denormal: add magic number and subtract
            float denorm = qx_pos + __uint_as_float(149u << 23);
            unsigned int d = __float_as_uint(denorm);
            d -= (149u << 23);
            e2m1 = (unsigned char)d;
        } else {
            // Normal: rounding with banker's rounding
            unsigned int mant_odd = (qx_bits >> 22) & 1;
            int val_to_add = ((1 - 127) << 23) + (1 << 21) - 1;
            int nx = (int)qx_bits + val_to_add + (int)mant_odd;
            nx = nx >> 22;
            e2m1 = (unsigned char)nx;
        }

        // Add sign back
        unsigned char sign_fp4 = (unsigned char)(sign >> 28);
        fp4_vals[i] = (e2m1 & 0x7) | sign_fp4;
    }

    // Pack fp4x2: evens in low nibble, odds in high nibble
    unsigned char* out = A_fp4 + m * (K / 2) + k_base / 2;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out[i] = (fp4_vals[2 * i] & 0xF) | ((fp4_vals[2 * i + 1] & 0xF) << 4);
    }

    // Store shuffled E8M0 scale
    int o0 = m / 32;
    int o1 = (m % 32) / 16;
    int o2 = m % 16;
    int o3 = qb / 8;
    int o4 = (qb % 8) / 4;
    int o5 = qb % 4;
    int sidx = o1 + o4 * 2 + o2 * 4 + o5 * 64 + o3 * 256 + o0 * 32 * scaleN_pad;
    A_scale[sidx] = (unsigned char)e8m0;
}

std::tuple<torch::Tensor, torch::Tensor> mxfp4_quant_hw(torch::Tensor A) {
    int M = A.size(0);
    int K = A.size(1);
    int num_qblocks = K / 32;
    int scaleN_pad = ((num_qblocks + 7) / 8) * 8;
    int M_pad = ((M + 255) / 256) * 256;

    auto A_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kUInt8).device(A.device()));
    auto A_scale = torch::zeros({M_pad, scaleN_pad},
        torch::dtype(torch::kUInt8).device(A.device()));

    int threads = min(num_qblocks, 256);
    int blocks_y = (num_qblocks + threads - 1) / threads;
    dim3 grid(M, blocks_y);
    dim3 block(threads);

    mxfp4_quant_kernel<<<grid, block>>>(
        (const unsigned short*)A.data_ptr(),
        (unsigned char*)A_fp4.data_ptr(),
        (unsigned char*)A_scale.data_ptr(),
        M, K, scaleN_pad
    );

    return std::make_tuple(A_fp4, A_scale);
}
'''

        cpp_src = r'''
#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor> mxfp4_quant_hw(torch::Tensor A);
'''

        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

        custom_kernel._module = load_inline(
            name='mxfp4_sw_quant',
            cpp_sources=[cpp_src],
            cuda_sources=[hip_src],
            extra_cuda_cflags=['-O3'],
            functions=['mxfp4_quant_hw'],
            verbose=False,
        )

    import aiter
    from aiter import dtypes

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()

    A_q, A_scale_sh = custom_kernel._module.mxfp4_quant_hw(A)

    out_gemm = aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2), B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
    return out_gemm
