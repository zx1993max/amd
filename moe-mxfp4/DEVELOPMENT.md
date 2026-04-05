# moe-mxfp4 开发记录

## 当前版本命名

只保留 `submission_v+数字.py`：

- `versions/submission_v1.py`
- `versions/submission_v2.py`
- `versions/submission_v3.py`

## 版本说明

- `v1`：双输入格式兼容（5 元组用于 mxfp4-mm，12 元组用于 moe-mxfp4），基线路径。
- `v2`：在 v1 基础上增加 `torch.inference_mode()` 与 `contiguous()` 规范化。
- `v3`：在 v1 基础上按 batch size 做自适应 contiguous。

## 兼容性目标

- 避免再次出现 `ValueError: expected 12, got 5`。
- 如果输入为 5 元组，自动走 mxfp4-mm 路径。
- 如果输入为 12 元组，自动走 moe-mxfp4 路径。

