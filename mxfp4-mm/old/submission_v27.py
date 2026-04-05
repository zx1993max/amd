from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    import torch
    import os

    if not hasattr(custom_kernel, '_module'):
        from torch.utils.cpp_extension import load_inline

        hip_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#if defined(__gfx950__)

// HW FP4 convert: fp4 = round(bf16 / scale_float)
// scale_float = 2^(e8m0-127) per CDNA4 ISA
__device__ __forceinline__ unsigned char cvt_fp4_byte(unsigned int pair, float sc) {
    unsigned int c = 0;
    asm volatile("v_cvt_scalef32_pk_fp4_bf16 %0, %1, %2"
        : "+v"(c) : "v"(pair), "v"(sc));
    return (unsigned char)(c & 0xFF);
}

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

    // Load 32 bf16, compute amax
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = bf16_to_f32(row[i]);
        float av = fabsf(v);
        if (av > amax) amax = av;
    }

    // E8M0 scale (same as v27b software version)
    unsigned int ai = __float_as_uint(amax);
    ai = (ai + 0x200000u) & 0xFF800000u;
    amax = __uint_as_float(ai);
    float log2_amax = log2f(amax);
    float scale_unbiased = floorf(log2_amax) - 2.0f;
    if (scale_unbiased < -127.0f) scale_unbiased = -127.0f;
    if (scale_unbiased > 127.0f) scale_unbiased = 127.0f;
    int e8m0 = (int)scale_unbiased + 127;
    if (e8m0 < 0) e8m0 = 0;
    if (e8m0 > 254) e8m0 = 254;

    // Scale for hw instruction: fp4 = round(bf16 / scale_float)
    // scale_float = bit_cast<float>(e8m0 << 23) = 2^(e8m0-127) = 2^(scale_unbiased)
    // Per CLAUDE.md: instruction DIVIDES by this value
    // Handle e8m0=0: use smallest denorm float to avoid div-by-zero
    unsigned int e8m0_bits = (e8m0 == 0) ? 0x00400000u : ((unsigned int)e8m0 << 23);
    float scf = __uint_as_float(e8m0_bits);

    // Load bf16 pairs as uint32 for hw instruction
    const unsigned int* row32 = (const unsigned int*)(A + m * K + k_base);

    // Convert using hw instruction - one byte at a time (aiter pattern)
    unsigned char* out = A_fp4 + m * (K / 2) + k_base / 2;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out[i] = cvt_fp4_byte(row32[i], scf);
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

#else
__global__ void mxfp4_quant_kernel(
    const unsigned short* __restrict__ A,
    unsigned char*        __restrict__ A_fp4,
    unsigned char*        __restrict__ A_scale,
    int M, int K, int scaleN_pad
) {}
#endif

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
            name='mxfp4_hw_quant_v2',
            cpp_sources=[cpp_src],
            cuda_sources=[hip_src],
            extra_cuda_cflags=['-O3', '-D__gfx950__'],
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
