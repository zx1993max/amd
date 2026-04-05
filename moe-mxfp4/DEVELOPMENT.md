# moe-mxfp4 开发记录

你说得对：这里所有工作都只围绕 **moe-mxfp4**。

## 当前版本命名

- `versions/submission_v1.py`
- `versions/submission_v2.py`
- `versions/submission_v3.py`

## 版本说明（仅 moe-mxfp4）

- `v1`：moe 基线，直接调用 `fused_moe`（shuffled weights/scales）。
- `v2`：在 v1 上加 `torch.inference_mode()` + 全量 contiguous。
- `v3`：在 v1 上加按 batch size 的自适应 contiguous（阈值 128）。

## 提交命令（moe）

```bash
popcorn-cli submit --mode benchmark --gpu MI355X --leaderboard amd-moe-mxfp4 moe-mxfp4/versions/submission_v1.py --no-tui
```

> 注意：不要再提交到 `amd-mxfp4-mm`，那是另一个题目。

