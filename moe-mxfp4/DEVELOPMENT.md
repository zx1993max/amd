# moe-mxfp4 开发记录

你说得对：这里所有工作都只围绕 **moe-mxfp4**。

## 当前版本命名

- `versions/submission_v1.py`
- `versions/submission_v2.py`
- `versions/submission_v3.py`
- `versions/submission_v4.py`

## 版本说明（仅 moe-mxfp4）

- `v1`：moe 基线，直接调用 `fused_moe`（shuffled weights/scales）。
- `v2`：在 v1 上加 `torch.inference_mode()` + 全量 contiguous。
- `v3`：在 v1 上加按 batch size 的自适应 contiguous（阈值 128）。
- `v4`：在 v1 上加“按输入指针和 shape 的输出缓存复用”。

## 提交命令（moe）

```bash
popcorn-cli submit --mode benchmark --gpu MI355X --leaderboard amd-moe-mxfp4 moe-mxfp4/versions/submission_v4.py --no-tui
```

> 注意：不要提交到 `amd-mxfp4-mm`，那是另一个题目。

## 借鉴 mxfp4-mm 的方法

- 强调版本化迭代与快速 AB 测试。
- 先基线稳定，再做“减少重复开销/减少额外操作”的策略。
- 保持最小化改动，快速提交、快速回传、快速迭代。

