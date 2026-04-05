# moe-mxfp4 初步开发梳理（对齐 mxfp4-mm 方法）

## 1) 项目现状

- 当前 `moe-mxfp4/submission.py` 为可直接提交的基线入口（AITER `fused_moe`，shuffled 权重路径）。
- `mxfp4-mm` 已形成“多版本 submission + 经验沉淀”的工作流，可直接复用到 `moe-mxfp4`。

## 2) 对 moe-mxfp4 的复用策略

1. **版本化迭代**：每个关键变更保存成独立 `submission_v*.py`，便于回滚和 AB 对比。
2. **最小顶层 import**：顶层只保留 `from task import ...`，其他依赖放到 `custom_kernel()`。
3. **先保正确、再压性能**：先用 AITER 保证 correctness，再做布局、调度与融合类优化。

## 3) 已准备的 3 个可测版本

- `versions/submission_v1_baseline_shuffled.py`
  - 纯 baseline：直接调用 `fused_moe` + shuffled weight/scale。
- `versions/submission_v2_inference_mode.py`
  - baseline + `torch.inference_mode()` + 核心输入 `.contiguous()` 规范化。
- `versions/submission_v3_adaptive_layout.py`
  - 按 batch size 自适应：大 batch (`bs>=128`) 走 contiguous 规范化路径。

> 说明：主入口 `moe-mxfp4/submission.py` 当前等价于 v1（便于默认提交）。

## 4) 目录使用约定

- 通用参考仓库：下载到 `extra_item/`
- `moe-mxfp4` 专项内容：下载到 `moe-mxfp4/`
- 所有迭代版提交：放在 `moe-mxfp4/versions/`

## 5) 测试反馈建议

你提交测试后，建议按以下格式反馈，方便我快速迭代：

- 版本名：`v1 / v2 / v3`
- test 是否通过：`pass/fail`
- benchmark 总体：`geom mean` 或各 case 延迟
- 若失败：报错栈 + 对应 case（shape/bs/topk）

