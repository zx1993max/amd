# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AMD MI355X (CDNA4, gfx950) kernel optimization competition with three problems: `mxfp4-mm`, `mixed-mla`, `moe-mxfp4`. Submissions are evaluated via `popcorn-cli` on remote MI355X hardware — no local GPU needed.

## Submission Commands

```bash
# Correctness test (10/hr limit)
popcorn-cli submit --mode test --gpu MI355X --leaderboard amd-mxfp4-mm mxfp4-mm/submission.py --no-tui

# Performance benchmark (6/hr limit)
popcorn-cli submit --mode benchmark --gpu MI355X --leaderboard amd-mxfp4-mm mxfp4-mm/submission.py --no-tui

# Leaderboard ranking (1/hr limit)
popcorn-cli submit --mode leaderboard --gpu MI355X --leaderboard amd-mxfp4-mm mxfp4-mm/submission.py --no-tui
```

Replace `amd-mxfp4-mm` with `amd-mixed-mla` or `amd-moe-mxfp4` for other problems.

## Architecture

**Evaluation flow:** `eval.py` → spawns subprocess → imports `submission.py:custom_kernel(data)` → times with CUDA events → compares against `reference.py:check_implementation(data, output)`.

Each problem directory contains:
- `task.yml` — problem spec, test/benchmark shapes, timeouts
- `task.py` — `input_t`/`output_t` type definitions
- `reference.py` — `generate_input()` and `check_implementation()` (or `ref_kernel()`)
- `submission.py` — your `custom_kernel(data) → output_t` implementation

Shared: `eval.py` (harness), `utils.py` (seed, tensor comparison, L2 cache flush).

**Leaderboard scoring:** geometric mean of benchmark latencies across all cases.

## Platform Constraints

The evaluation platform monitors GPU stream ownership. Key rules:

**Code must NOT contain the string "stream"** — detected by grep (variables, comments, anything).

**Safe patterns:**
- Top-level: only `from task import input_t, output_t`
- All other imports inside `custom_kernel()` function body
- `aiter.ops.triton.quant.dynamic_mxfp4_quant` / `aiter.gemm_a4w4` / `aiter.utility.fp4_utils.e8m0_shuffle`
- File system operations (aiter source files at `/home/runner/aiter/` are writable)

**Triggers 500 error:**
- `torch.cuda.CUDAGraph()`, `torch.cuda.synchronize()`
- `aiter.gemm_a4w4_asm` (low-level ASM interface)
- `ctypes + hipModuleLoad`, `tinygrad` (officially banned)
- Top-level `import triton` (causes GPU init)
- Direct execution of custom `@triton.jit` kernels

**Source injection technique:** Overwrite aiter Triton kernel source files before `import aiter`, clear `__pycache__`. Platform treats the modified code as legitimate aiter.

## MXFP4-MM Problem

**Flow:** bf16 A → quantize to MXFP4 (per 1×32 block) → `gemm_a4w4` with pre-quantized B → bf16 C

**Input tuple:** `(A[m,k] bf16, B[n,k] bf16, B_q[n,k/2] fp4x2, B_shuffle[n,k/2] fp4x2, B_scale_sh[*,k/32] e8m0)`

**Performance breakdown:** quantization ~10μs + shuffle ~2μs + GEMM ~8μs ≈ 20μs baseline. GEMM is near hardware limit; optimization space is in quantization.

**Hardware FP4 instruction:** `v_cvt_scalef32_pk_fp4_bf16` converts 2 bf16→packed fp4 in one instruction. Available via `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(old_u32, bf16x2, scale, byte_sel)`.

## Environment

- PyTorch 2.10.0+rocm7.1, Triton 3.6.0, Python 3.12
- `aiter` library at `/home/runner/aiter/` (writable source)
- Docker: `gpu-mode/kernelbot/docker/amd-docker.Dockerfile`
- 420s test/benchmark timeout, 600s leaderboard timeout

## Key Files in extra_item/

- `skill.md` — comprehensive technical constraints, version history, API signatures
- `开发清单.md` — development roadmap and optimization strategies
- `aiter_repo/` — cloned aiter source for reference (hardware instruction examples in `csrc/include/ck_tile/vec_convert.h` and `csrc/include/aiter_opus_plus.h`)
