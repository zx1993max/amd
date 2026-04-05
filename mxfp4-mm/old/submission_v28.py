from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()

    # Original aiter quant (no source injection) + ASM GEMM
    A_fp4, A_bs = dynamic_mxfp4_quant(A)
    A_bs = e8m0_shuffle(A_bs)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = A_bs.view(dtypes.fp8_e8m0)

    out = aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
    return out
