# AMD MI355X MXFP4-MM 技能库（更新版，至 v51 / v52 plan）

更新日期：2026-04-05

## 一句话现状

当前稳定主线不是继续重构，而是：

- 以 **v48_fixed / v49 档位** 作为稳态基线
- M16 维持已打穿的 exact-reduce 路线
- M64 做低风险单变量微调
- M256 继续尝试**结构性突破**
- 但结构性突破的方向，已经从“改读法/删重排”收敛为：  
  **控资源压力，而不是改变 global-memory 读法**

## 当前最重要的结论

### 已证实有效
1. **v43** 证明 exact-shape specialization 是对的。
2. **v45** 证明 M16 的突破点在 exact reduce，不在 atomic。
3. **v47b** 证明 M16 exact compute 里 `b / b_scales` 的 `.cg` 是真增益。
4. **v48_fixed / v49** 证明当前主线可以在不打坏全局的情况下稳定运行，M16 稳在 10µs 内，M64/M256 维持竞争档位。fileciteturn19file0turn20file0

### 已高效证伪
1. **v50**：把 M256 的 `b` 改成 final-layout direct-gather，M256 从 ~12.3–12.6µs 退到 ~15.4–15.8µs。  
   结论：**不能破坏当前连续块读取。** fileciteturn21file0
2. **v50b**：只把 `b_scales` 改成 direct-gather，M256 更差到 ~17.4–17.7µs。  
   结论：**`b_scales` 也不能轻易改成 global-memory gather。** fileciteturn22file0
3. **v51**：保留连续块读取，但把 6 次 K-iter 完全展开，M256 仍在 ~16.4–16.5µs。  
   结论：**问题不主要在循环控制本身，而更像在寄存器压力 / live range / occupancy。** fileciteturn23file0

## 现在对 M256 的核心认识

M256 这组当前不该继续赌的方向：

- `b` direct-gather
- `b_scales` direct-gather
- full-unroll exact kernel

这三条都已被结果否定。

现在最像真突破口的判断是：

> **M256 的下一次结构性提升，必须从“控资源压力”出发，而不是继续从“改读法/删重排”出发。**

更具体说：

- 保住当前连续块读取
- 保住 block-local 重排
- 不再试 final-layout gather
- 重点从下面两件事里找突破：
  1. **半展开 + 降 live range**
  2. **128 宽子 tile 的 exact 结构**

## 版本演化速记

### v42
- 试图在热路径里做新的 scale 预排布
- 全线退化
- 教训：**不能在 hot path 里 materialize 新 tensor**

### v43
- exact-shape kernel 建立
- 三个大 shape 首次明显下降
- geomean 进入 ~9.6µs 档

### v44
- M16 atomic reduce + 重调 M64/M256
- 明显退化
- 教训：**M16 不能用 atomic；M64/M256 骨架别乱动**

### v45
- M16 exact compute + exact reduce
- M16 从 ~16.1µs 直接到 ~10.8µs
- geomean 到 ~8.90µs

### v46 / v47a / v47b
- 拆分验证后确认：
- **`.cg` 是 M16 的真收益**
- `reduce 64→128` 是假收益
- 新基线变成 **v47b**

### v48_fixed / v49
- 低风险单变量微调，整体不坏
- M256 的 A-side `.cg` 没形成结构性突破
- 结论：M256 下一刀不能停留在 cache hint 级别

### v50 / v50b / v51
- 三次高质量失败
- 成功把三条错误结构分支关掉
- 搜索空间显著缩小

## 当前推荐主线

- **稳态主线基线**：`v48_fixed / v49`
- **结构实验主线**：只围绕 M256 做资源压力控制
- **M16**：不再大改
- **M64**：只做极少数低风险小调
- **K=512 三组**：只做顺手收尾，不当主战场

## v52 的设计意图

v52 是对 “128 宽子 tile 的 exact 结构” 的第一次明确尝试：

- 不再改 M256 的读法
- 不再做 direct-gather
- 不再 full-unroll
- 直接把 M256 exact kernel 从 **16x256** 改成 **16x128** 子 tile 结构
- 目的：**半掉 accumulator 宽度，直接打寄存器/资源压力**

如果 v52 仍然不行，那么下一步更像该回到：

- `16x256` 结构不动
- 改成 **2x3 或 3x2 半展开**，只压 live range

## 当前最重要的工程规则

1. 失败不可怕，但失败必须能缩小搜索空间
2. 不再 bundle 多个结构性想法一起测
3. M256 的实验必须一次只打一刀
4. 任何新实验都要能回答明确问题：
   - 是读法问题？
   - 是解释成本问题？
   - 还是资源压力问题？

## 当前最值得记住的一句话

> 我们不是没找到方向，而是已经明确知道三条错路了；  
> 接下来最该赌的，是 **“控资源压力”**，而不是继续赌 **“改读法/删重排”**。
