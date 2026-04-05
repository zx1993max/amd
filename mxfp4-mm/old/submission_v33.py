import os
import shutil
import sys
import json

# ==========================================
# 1. Source Inject: quant kernel
# ==========================================
KERNEL_FILE = '/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/quant.py'
WRAPPER_FILE = '/home/runner/aiter/aiter/ops/triton/quant/quant.py'

KERNEL_CODE = r"""# SPDX-License-Identifier: MIT
import triton
import triton.language as tl

@triton.jit
def _static_per_tensor_quant_fp8_i8_kernel(
    qx_ptr, x_in_ptr, scale_in_ptr, cols: int, x_in_stride_r: int, NUM_COL_POW2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)
    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")
    scale = tl.load(scale_in_ptr)
    qx = (x / scale).to(qx_ptr.dtype.element_ty)
    tl.store(qx_ptr + offs, qx, mask=mask)

@triton.jit
def _dynamic_per_tensor_quant_fp8_i8_kernel(
    x_in_ptr, scale_out_ptr, cols: int, x_in_stride_r: int, NUM_COL_POW2: tl.constexpr, DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)
    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")
    m = tl.max(tl.abs(x))
    tl.atomic_max(scale_out_ptr, m / DTYPE_MAX, sem="relaxed")

@triton.jit
def _dynamic_per_token_quant_fp8_i8_kernel(
    qx_ptr, scale_out_ptr, x_in_ptr, cols: int, x_in_stride_r: int, NUM_COL_POW2: tl.constexpr, DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)
    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")
    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    qx = (x / scale_out).to(qx_ptr.dtype.element_ty)
    tl.store(scale_out_ptr + pid, scale_out)
    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")

@triton.jit
def _mxfp4_quant_op(x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE):
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1
    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
    quant_scale = tl.exp2(-scale_e8m0_unbiased)
    qx = x * quant_scale
    qx = qx.to(tl.uint32, bitcast=True)
    s = qx & 0x80000000
    qx = qx ^ s
    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)
    denorm_exp: tl.constexpr = (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.int32, bitcast=True)
    denormal_x = denormal_x - denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)
    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x_int = normal_x.to(tl.int32, bitcast=True)
    normal_x_int = normal_x_int + val_to_add + mant_odd.to(tl.int32)
    normal_x = normal_x_int.to(tl.uint32, bitcast=True)
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(e2m1_value,[BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2])
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)
    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

@triton.heuristics({"EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0 and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0})
@triton.jit
def _dynamic_mxfp4_quant_kernel(
    x_ptr, x_fp4_ptr, bs_ptr, stride_x_m_in, stride_x_n_in, stride_x_fp4_m_in, stride_x_fp4_n_in,
    stride_bs_m_in, stride_bs_n_in, M, N, scaleN,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, NUM_ITER: tl.constexpr, NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr, SHUFFLE_SCALE: tl.constexpr, EVEN_M_N: tl.constexpr, SCALING_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(tl.float32)
        out_tensor, bs_e8m0 = _mxfp4_quant_op(x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE)
        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)
        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        if SHUFFLE_SCALE:
            o0 = bs_offs_m // 32
            o1 = (bs_offs_m % 32) // 16
            o2 = bs_offs_m % 16
            o3 = bs_offs_n // 8
            o4 = (bs_offs_n % 8) // 4
            o5 = bs_offs_n % 4
            shuffled_idx = (o1[:, None] + o4[None, :] * 2 + o2[:, None] * 4 +
                o5[None, :] * 64 + o3[None, :] * 256 + o0[:, None] * 32 * scaleN)
            if EVEN_M_N:
                tl.store(bs_ptr + shuffled_idx, bs_e8m0)
            else:
                bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
                tl.store(bs_ptr + shuffled_idx, bs_e8m0, mask=bs_mask)
        else:
            bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
            if EVEN_M_N:
                tl.store(bs_ptr + bs_offs, bs_e8m0)
            else:
                bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
                tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)
"""

WRAPPER_CODE = r"""# SPDX-License-Identifier: MIT
import torch
import triton
from .._triton_kernels.quant.quant import (
    _dynamic_mxfp4_quant_kernel, _mxfp4_quant_op,
    _static_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_token_quant_fp8_i8_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
def static_per_tensor_quant_fp8_i8(qx, x_in, scale_in):
    rows, cols = x_in.shape
    _static_per_tensor_quant_fp8_i8_kernel[(rows,)](qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=triton.next_power_of_2(cols))
    return qx
def dynamic_per_tensor_quant_fp8_i8(qx, x_in, scale_out):
    rows, cols = x_in.shape
    p2 = triton.next_power_of_2(cols)
    dm = torch.finfo(qx.dtype).max if torch.is_floating_point(qx) else torch.iinfo(qx.dtype).max
    _dynamic_per_tensor_quant_fp8_i8_kernel[(rows,)](x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=p2, DTYPE_MAX=dm)
    _static_per_tensor_quant_fp8_i8_kernel[(rows,)](qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=p2)
    return qx, scale_out
def dynamic_per_token_quant_fp8_i8(qx, x_in, scale_out):
    rows, cols = x_in.shape
    p2 = triton.next_power_of_2(cols)
    dm = torch.finfo(qx.dtype).max if torch.is_floating_point(qx) else torch.iinfo(qx.dtype).max
    _dynamic_per_token_quant_fp8_i8_kernel[(rows,)](qx, scale_out, x_in, cols, x_in.stride(0), NUM_COL_POW2=p2, DTYPE_MAX=dm)
    return qx, scale_out
def dynamic_mxfp4_quant(x, shuffle=False):
    M, N = x.shape
    MXFP4_QUANT_BLOCK_SIZE = 32
    assert (N // 2) % 2 == 0
    scaleN_raw = (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    if M <= 32:
        NUM_ITER, BLOCK_SIZE_M, BLOCK_SIZE_N = 1, triton.next_power_of_2(M), 32
        NUM_WARPS, NUM_STAGES = 1, 1
    else:
        NUM_ITER, BLOCK_SIZE_M, BLOCK_SIZE_N = 4, 64, 64
        NUM_WARPS, NUM_STAGES = 4, 2
        if N <= 16384: BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 128
    if N <= 1024:
        NUM_ITER, NUM_STAGES, NUM_WARPS = 1, 1, 4
        BLOCK_SIZE_N = max(32, min(256, triton.next_power_of_2(N)))
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    if shuffle:
        M_pad = (M + 255) // 256 * 256
        scaleN_pad = (scaleN_raw + 7) // 8 * 8
        bs_e8m0 = torch.zeros((M_pad * scaleN_pad,), dtype=torch.uint8, device=x.device)
    else:
        scaleN_pad = scaleN_raw
        bs_e8m0 = torch.empty((scaleN_raw, M), dtype=torch.uint8, device=x.device).T
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER))
    _dynamic_mxfp4_quant_kernel[grid](
        x, x_fp4, bs_e8m0, x.stride(0), x.stride(1), x_fp4.stride(0), x_fp4.stride(1),
        bs_e8m0.stride(0) if not shuffle else 0, bs_e8m0.stride(1) if not shuffle else 0,
        M, N, scaleN_pad, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_ITER=NUM_ITER, NUM_STAGES=NUM_STAGES, MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SHUFFLE_SCALE=shuffle, SCALING_MODE=0, num_warps=NUM_WARPS, waves_per_eu=0, num_stages=1)
    if shuffle: bs_e8m0 = bs_e8m0.view(M_pad, scaleN_pad)
    return x_fp4, bs_e8m0
"""

# ==========================================
# 2. Source inject: preshuffle kernel — add ATOMIC_ADD support
# ==========================================
PRESHUFFLE_KERNEL_FILE = '/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py'
PRESHUFFLE_WRAPPER_FILE = '/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py'

# Patch 1: Add ATOMIC_ADD to preshuffle kernel parameter list
KERNEL_PATCH_OLD = """    PREQUANT: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    \\"\\"\\"Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format."""

KERNEL_PATCH_NEW = """    PREQUANT: tl.constexpr,
    cache_modifier: tl.constexpr,
    ATOMIC_ADD: tl.constexpr = False,
):
    \\"\\"\\"Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format."""

# Patch 2: Replace store section to support atomic_add
STORE_PATCH_OLD = """        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)"""

STORE_PATCH_NEW = """        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if ATOMIC_ADD:
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            tl.atomic_add(c_ptrs, accumulator.to(c_ptr.type.element_ty), mask=c_mask)
        else:
            c = accumulator.to(c_ptr.type.element_ty)
            c_ptrs = (
                c_ptr
                + stride_cm * offs_cm[:, None]
                + stride_cn * offs_cn[None, :]
                + pid_k * stride_ck
            )
            tl.store(c_ptrs, c, mask=c_mask)"""

# ==========================================
# 3. Source inject: preshuffle wrapper — add atomic_add path
# ==========================================

# Patch wrapper: replace the reduce section with atomic_add support
WRAPPER_PATCH_OLD = """    if return_y_pp:
        return y_pp
    elif config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 64"""

WRAPPER_PATCH_NEW = """    if return_y_pp:
        return y_pp
    elif config.get("_ATOMIC_ADD"):
        # atomic_add mode: output already accumulated, just return
        return y
    elif config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 64"""

# Patch wrapper: pass ATOMIC_ADD to kernel call
WRAPPER_KERNEL_CALL_OLD = """        PREQUANT=prequant,
        **config,
    )"""

WRAPPER_KERNEL_CALL_NEW = """        PREQUANT=prequant,
        ATOMIC_ADD=config.pop("_ATOMIC_ADD", False),
        **config,
    )"""

# ==========================================
# 4. Inject config files
# ==========================================
CONFIGS_DIR = '/home/runner/aiter/aiter/ops/triton/configs/gemm'

SHAPE_CONFIGS = {
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2880-K=512.json": {
        "M_LEQ_8": {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "M_LEQ_32": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
    },
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=4096-K=512.json": {
        "M_LEQ_8": {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "M_LEQ_32": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
    },
    # N=2112,K=7168 already exists with NUM_KSPLIT=14
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=7168-K=2048.json": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 4},
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 1},
    },
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=3072-K=1536.json": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 3},
        "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 1},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 1},
    },
}

# ==========================================
# 5. Execute all injections
# ==========================================
try:
    # Write quant files
    with open(KERNEL_FILE, 'w') as f:
        f.write(KERNEL_CODE)
    with open(WRAPPER_FILE, 'w') as f:
        f.write(WRAPPER_CODE)

    # Write config files
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    for fname, cfg in SHAPE_CONFIGS.items():
        with open(os.path.join(CONFIGS_DIR, fname), 'w') as f:
            json.dump(cfg, f, indent=2)

    # Patch preshuffle kernel: add ATOMIC_ADD
    with open(PRESHUFFLE_KERNEL_FILE, 'r') as f:
        ksrc = f.read()
    ksrc = ksrc.replace(
        '    PREQUANT: tl.constexpr,\n    cache_modifier: tl.constexpr,\n):\n    """Kernel for computing the matmul C = A x B.\n    A and B inputs are in the microscale fp4 (mxfp4) format.',
        '    PREQUANT: tl.constexpr,\n    cache_modifier: tl.constexpr,\n    ATOMIC_ADD: tl.constexpr = False,\n):\n    """Kernel for computing the matmul C = A x B.\n    A and B inputs are in the microscale fp4 (mxfp4) format.'
    )
    ksrc = ksrc.replace(STORE_PATCH_OLD, STORE_PATCH_NEW)
    with open(PRESHUFFLE_KERNEL_FILE, 'w') as f:
        f.write(ksrc)

    # Patch preshuffle wrapper: add atomic_add path
    with open(PRESHUFFLE_WRAPPER_FILE, 'r') as f:
        wsrc = f.read()
    wsrc = wsrc.replace(WRAPPER_PATCH_OLD, WRAPPER_PATCH_NEW)
    wsrc = wsrc.replace(WRAPPER_KERNEL_CALL_OLD, WRAPPER_KERNEL_CALL_NEW)
    with open(PRESHUFFLE_WRAPPER_FILE, 'w') as f:
        f.write(wsrc)
except Exception:
    pass

# Clear ALL caches
for d in [
    '/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/__pycache__',
    '/home/runner/aiter/aiter/ops/triton/quant/__pycache__',
    '/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/__pycache__',
    '/home/runner/aiter/aiter/ops/triton/gemm/basic/__pycache__',
]:
    if os.path.exists(d):
        try: shutil.rmtree(d)
        except Exception: pass

# ==========================================
# 6. Entry point — aggressive shape router
# ==========================================
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    import torch
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]
    A = A.contiguous()

    # Convert B formats (views, near-zero cost)
    w = B_shuffle.view(torch.uint8).reshape(N // 16, -1)
    bs = B_scale_sh.view(torch.uint8)
    N_pad, scaleN_pad = bs.shape
    w_scales = bs.reshape(N_pad // 32, 32 * scaleN_pad)[:N // 32, :K]

    # ===== Shape-specific dispatch =====

    if K > 2048:
        # M=16, K=7168, N=2112: preshuffle + atomic_add (NUM_KSPLIT=14)
        # Saves ~2μs by eliminating reduce kernel entirely
        y = torch.zeros((M, N), dtype=torch.bfloat16, device=A.device)
        cfg = {
            "BLOCK_SIZE_M": 16 if M > 8 else 8,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 1,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 14,
            "_ATOMIC_ADD": True,
        }
        try:
            return gemm_a16wfp4_preshuffle(A, w, w_scales, config=cfg, dtype=torch.bfloat16, y=y)
        except Exception:
            # Fallback: skip_reduce + torch.sum (v32 proven approach)
            y_pp = gemm_a16wfp4_preshuffle(A, w, w_scales, dtype=torch.bfloat16, skip_reduce=True)
            return y_pp.sum(dim=0).to(torch.bfloat16)

    elif M >= 128:
        # M=256, K=1536: ASM path (18.1μs, preshuffle is 20.7μs)
        A_q, A_scale_sh = dynamic_mxfp4_quant(A, shuffle=True)
        return aiter.gemm_a4w4(
            A_q.view(dtypes.fp4x2), B_shuffle,
            A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
            dtype=dtypes.bf16, bpreshuffle=True,
        )

    elif K > 512:
        # M=64, K=2048: preshuffle + try NUM_KSPLIT=4 with atomic_add
        y = torch.zeros((M, N), dtype=torch.bfloat16, device=A.device)
        cfg = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 4,
            "_ATOMIC_ADD": True,
        }
        try:
            return gemm_a16wfp4_preshuffle(A, w, w_scales, config=cfg, dtype=torch.bfloat16, y=y)
        except Exception:
            # Fallback: preshuffle without split
            return gemm_a16wfp4_preshuffle(A, w, w_scales, dtype=torch.bfloat16)

    else:
        # K=512: preshuffle, NUM_KSPLIT=1 (auto config from JSON)
        return gemm_a16wfp4_preshuffle(A, w, w_scales, dtype=torch.bfloat16)
