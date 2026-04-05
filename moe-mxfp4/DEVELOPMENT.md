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

## 版本说明（仅 moe-mxfp4）

- `v1`：moe 基线，直接调用 `fused_moe`（shuffled weights/scales）。
- `v2`：在 v1 上加 `torch.inference_mode()` + 全量 contiguous。
- `v3`：在 v1 上加按 batch size 的自适应 contiguous（阈值 128）。
- `v4`：输出缓存复用（被 KernelGuard `LAST_CALL_REPLAY` 拒绝，不再用于提交）。
- `v5`：实验 `expert_mask` 预分配缓存（仅缓存全 1 mask，避免每次创建）。

## 直接在 main 开发（不走 PR）

如果你不想每次提 PR，需要仓库管理员在 GitHub 关闭/放宽以下保护策略：

1. Branch protection（`main`）里关闭 **Require a pull request before merging**。
2. 关闭或放宽 required status checks / required reviews / restrict who can push。
3. 允许你账号对 `main` 直接 push。

本地建议配置（减少 pull 冲突）：

```bash
git config pull.rebase true
git config rebase.autoStash true
git config fetch.prune true
```

日常流程：

```bash
git checkout main
git pull --rebase origin main
# 修改代码
git add -A
git commit -m "..."
git push origin main
```

## 提交命令（moe）

```bash
popcorn-cli submit --mode benchmark --gpu MI355X --leaderboard amd-moe-mxfp4 moe-mxfp4/versions/submission_v5.py --no-tui
```

> 注意：不要提交到 `amd-mxfp4-mm`，那是另一个题目。

## 借鉴 mxfp4-mm 的方法

- 强调版本化迭代与快速 AB 测试。
- 先基线稳定，再做“减少重复开销/减少额外操作”的策略。
- 保持最小化改动，快速提交、快速回传、快速迭代。

