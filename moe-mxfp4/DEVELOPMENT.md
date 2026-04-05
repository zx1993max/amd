# moe-mxfp4 开发记录

你说得对：这里所有工作都只围绕 **moe-mxfp4**。

## 版本冻结规则（重要）

- 历史版本一旦产出（如 `submission_v1.py`、`submission_v2.py`），默认**冻结不改**。
- 如果发现历史版本有问题，不在原文件上改，新增修复版：`submission_vX_fix.py`。
- 迭代优化同样走新版本：`submission_v{N+1}.py`。

## 当前版本命名

- `versions/submission_v1.py`
- `versions/submission_v2.py`
- `versions/submission_v3.py`
- `versions/submission_v4.py`
- `versions/submission_v5.py`
- `versions/submission_v5_fix.py`
- `versions/submission_v6.py`

## 版本说明（仅 moe-mxfp4）

- `v1`：moe 基线，直接调用 `fused_moe`（shuffled weights/scales）。
- `v2`：在 v1 上加 `torch.inference_mode()` + 全量 contiguous。
- `v3`：在 v1 上加按 batch size 的自适应 contiguous（阈值 128）。
- `v4`：输出缓存复用（被 KernelGuard `LAST_CALL_REPLAY` 拒绝，不再用于提交）。
- `v5`：实验 `expert_mask` 缓存（线上出现 memory fault，已废弃）。
- `v5_fix`：回退到稳定逻辑（等价 v3 路线），用于安全提交。
- `v6`：实验 `doweight_stage1=True`，尝试触发更优 kernel 路径。

## 推荐提交顺序（时间紧）

1. 先提 `v5_fix`（确保稳定通过）
2. 再提 `v6`（观察是否有显著提速）
3. 若 `v6` 精度/稳定异常，立即回退 `v5_fix`

## 提交命令（moe）

```bash
popcorn-cli submit --mode benchmark --gpu MI355X --leaderboard amd-moe-mxfp4 moe-mxfp4/versions/submission_v5_fix.py --no-tui
popcorn-cli submit --mode benchmark --gpu MI355X --leaderboard amd-moe-mxfp4 moe-mxfp4/versions/submission_v6.py --no-tui
```

> 注意：不要提交到 `amd-mxfp4-mm`，那是另一个题目。

