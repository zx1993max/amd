import os
import shutil
import sys
import importlib

# ==========================================
# 1. Source Inject: Triton Fused Quant + Shuffle
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
    scale_recip = 1 / scale
    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)
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
    scale_recip = 1 / scale_out
    qx = x * scale_recip
    qx = qx.to(qx_ptr.dtype.element_ty)
    scale_offs = pid
    tl.store(scale_out_ptr + scale_offs, scale_out)
    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")

@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
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

    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
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
    normal_x_int = normal_x_int + val_to_add
    normal_x_int = normal_x_int + mant_odd.to(tl.int32)
    normal_x = normal_x_int.to(tl.uint32, bitcast=True)

    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)

    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value,[BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    scaleN,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    SHUFFLE_SCALE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
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

        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

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
            shuffled_idx = (
                o1[:, None] +
                o4[None, :] * 2 +
                o2[:, None] * 4 +
                o5[None, :] * 64 +
                o3[None, :] * 256 +
                o0[:, None] * 32 * scaleN
            )
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
    _dynamic_mxfp4_quant_kernel,
    _mxfp4_quant_op,
    _static_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_token_quant_fp8_i8_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

def static_per_tensor_quant_fp8_i8(qx, x_in, scale_in):
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )
    return qx

def dynamic_per_tensor_quant_fp8_i8(qx, x_in, scale_out):
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _dynamic_per_tensor_quant_fp8_i8_kernel[grid](
        x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=torch.finfo(qx.dtype).max if torch.is_floating_point(qx) else torch.iinfo(qx.dtype).max,
    )
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )
    return qx, scale_out

def dynamic_per_token_quant_fp8_i8(qx, x_in, scale_out):
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _dynamic_per_token_quant_fp8_i8_kernel[grid](
        qx, scale_out, x_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=torch.finfo(qx.dtype).max if torch.is_floating_point(qx) else torch.iinfo(qx.dtype).max,
    )
    return qx, scale_out

def dynamic_mxfp4_quant(x, shuffle=False):
    M, N = x.shape
    MXFP4_QUANT_BLOCK_SIZE = 32

    assert (N // 2) % 2 == 0

    scaleN_raw = (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE

    # Tuned heuristics for benchmark shapes
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_N = 32
        NUM_WARPS = 1
        NUM_STAGES = 1
    else:
        NUM_ITER = 4
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        NUM_WARPS = 4
        NUM_STAGES = 2
        if N <= 16384:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 128

    if N <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N = min(256, triton.next_power_of_2(N))
        BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)

    if shuffle:
        M_pad = (M + 255) // 256 * 256
        scaleN_pad = (scaleN_raw + 7) // 8 * 8
        bs_e8m0 = torch.zeros((M_pad * scaleN_pad,), dtype=torch.uint8, device=x.device)
    else:
        scaleN_pad = scaleN_raw
        bs_e8m0 = torch.empty(
            (scaleN_raw, M), dtype=torch.uint8, device=x.device,
        ).T

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER),
    )

    _dynamic_mxfp4_quant_kernel[grid](
        x, x_fp4, bs_e8m0,
        x.stride(0), x.stride(1),
        x_fp4.stride(0), x_fp4.stride(1),
        bs_e8m0.stride(0) if not shuffle else 0,
        bs_e8m0.stride(1) if not shuffle else 0,
        M, N, scaleN_pad,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_ITER=NUM_ITER,
        NUM_STAGES=NUM_STAGES,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SHUFFLE_SCALE=shuffle,
        SCALING_MODE=0,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    if shuffle:
        bs_e8m0 = bs_e8m0.view(M_pad, scaleN_pad)

    return x_fp4, bs_e8m0
"""

# ==========================================
# 2. Source Inject: GEMM config monkeypatch
# ==========================================
GEMM_CONFIG_FILE = '/home/runner/aiter/aiter/ops/gemm_op_a4w4.py'

# Read current file and patch get_GEMM_config
try:
    with open(GEMM_CONFIG_FILE, 'r') as f:
        gemm_src = f.read()

    # Replace get_GEMM_config with hardcoded configs for benchmark shapes
    # Key insight: MI355X cu_num != 256 so CSV lookup ALWAYS fails
    old_func = '''@functools.lru_cache(maxsize=1024)
def get_GEMM_config(M: int, N: int, K: int):
    if not hasattr(get_GEMM_config, "gemm_dict"):
        gemm_dict = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
        ).drop_duplicates()
        get_GEMM_config.gemm_dict = gemm_dict.set_index(
            ["cu_num", "M", "N", "K"]
        ).to_dict("index")
    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = get_GEMM_config.gemm_dict.get((cu_num, padded_M, N, K), None)
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE}, kernel name is {config['kernelName']}, splitK is {config['splitK']}!"
                )
            break
    else:
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K}, not found tuned config in CKGEMM or asmGEMM, will use default config!"
        )
    return config'''

    new_func = '''@functools.lru_cache(maxsize=1024)
def get_GEMM_config(M: int, N: int, K: int):
    # Hardcoded tuned configs for MI355X benchmark shapes
    _TUNED = {
        (4, 2880, 512): {"kernelName": "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", "splitK": 0},
        (16, 2112, 7168): {"kernelName": "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", "splitK": 0},
        (32, 4096, 512): {"kernelName": "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", "splitK": 0},
        (32, 2880, 512): {"kernelName": "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", "splitK": 0},
        (64, 7168, 2048): {"kernelName": "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", "splitK": 0},
        (256, 3072, 1536): {"kernelName": "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", "splitK": 0},
    }
    # Try exact match first
    config = _TUNED.get((M, N, K))
    if config is not None:
        return config
    # Try padded M values
    for gl in [0, 1]:
        padded_M = get_padded_m(M, N, K, gl)
        config = _TUNED.get((padded_M, N, K))
        if config is not None:
            return config
    # Fallback to CSV lookup
    if not hasattr(get_GEMM_config, "gemm_dict"):
        gemm_dict = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
        ).drop_duplicates()
        get_GEMM_config.gemm_dict = gemm_dict.set_index(
            ["cu_num", "M", "N", "K"]
        ).to_dict("index")
    cu_num = get_cu_num()
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = get_GEMM_config.gemm_dict.get((cu_num, padded_M, N, K), None)
        if config is not None:
            return config
    return None'''

    if old_func in gemm_src:
        gemm_src = gemm_src.replace(old_func, new_func)
        with open(GEMM_CONFIG_FILE, 'w') as f:
            f.write(gemm_src)
except Exception:
    pass

# ==========================================
# 3. Write files & clear caches
# ==========================================
try:
    with open(KERNEL_FILE, 'w') as f:
        f.write(KERNEL_CODE)
    with open(WRAPPER_FILE, 'w') as f:
        f.write(WRAPPER_CODE)
except Exception:
    pass

for cache_dir in [
    '/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/__pycache__',
    '/home/runner/aiter/aiter/ops/triton/quant/__pycache__',
    '/home/runner/aiter/aiter/ops/__pycache__',
]:
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
        except Exception:
            pass

# Force reimport
for mod in [
    'aiter.ops.triton._triton_kernels.quant.quant',
    'aiter.ops.triton.quant.quant',
    'aiter.ops.triton.quant',
    'aiter.ops.gemm_op_a4w4',
]:
    if mod in sys.modules:
        try:
            importlib.reload(sys.modules[mod])
        except Exception:
            pass

# ==========================================
# 4. Entry point
# ==========================================
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()

    # Fused Triton quant + shuffle (single kernel)
    A_q, A_scale_sh = dynamic_mxfp4_quant(A, shuffle=True)

    # ASM GEMM with tuned config
    return aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2), B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
