# AMD MI355X MXFP4-MM 技能库（更新版，至 v59）

> 更新日期：2026-04-05  
> 目标：在 **合法、可复现、可提交** 的前提下，把当前主线继续压向 **geo mean 8.2µs** 附近。  
> 当前主线：**v55**  
> 当前状态：**v59 已制作，待复测；v56/v57/v58 已形成高价值负结果，帮助收敛结构方向。**

---

## 一、当前一句话结论

这条主线已经非常清楚：

- **v43** 证明：`exact-shape specialization` 是真正的结构收益来源
- **v45** 证明：M16 的成功不是因为 split 本身，而是因为 **exact reduce** 替掉了错误的 reduce 方式
- **v49 / v55** 证明：对 M64/M256，真正有效的是 **在当前最快主家族内控资源压力**，而不是重开新家族
- **v56** 证明：不能把 M16 的 `split-K + exact reduce` 经验直接套到 M64/M256
- **v57** 证明：wrapper / Python dispatch 不是 M64/M256 的主瓶颈
- **v58** 证明：把 M64 切到 `dynamic_mxfp4_quant + aiter.gemm_a4w4(bpreshuffle=True)` 这种 **分离 quant + gemm family** 是错误方向
- **v59** 是第一次真正只攻 **current family 内部 kernel 热点静态化** 的版本，目标是验证 M64 的热点是否在 `b_scales` 局部 layout 解释

压成一句话：

> **M16 的成功经验要抽象成“找出最贵的通用逻辑，再只对那个位置 exact 化”，而不是机械地套用 split/reduce。**
>  
> **对 M64/M256，当前最值得继续押的是 current fused family 内部的 kernel 本体，而不是 wrapper、family switch 或泛化 split/reduce。**

---

## 二、平台规则与安全边界

### 1. 已确认安全/可用

- 顶层只保留 `from task import input_t, output_t`
- 其他 import 全放函数内
- `aiter.ops.triton.quant.dynamic_mxfp4_quant` 可用
- `aiter.utility.fp4_utils.e8m0_shuffle` 可用
- `aiter.gemm_a4w4` 高层封装可用
- 函数内做 `os/shutil/json` 文件系统操作可用
- 在 import aiter 前覆写 `/home/runner/aiter/...` 下源码文件是可行的
- `torch.utils.cpp_extension.load_inline()` 现在看是可用的，但目前没有证明它比 Triton 注入主线更快

### 2. 已确认危险/不该碰

- `torch.cuda.CUDAGraph()`
- `torch.cuda.synchronize()`
- 顶层 `import triton`
- 自定义 Triton kernel 直接 launch
- `aiter.utility.fp4_utils.dynamic_mxfp4_quant(shuffle=True)`
- `aiter.gemm_a4w4_asm` 直接裸调
- 任何看起来像 exploit / cache hack / skip compute / clamp compute 的方案

### 3. 当前合规原则

- 必须是 **production-worthy** 路线
- 不做 tolerance exploit
- 不做跨迭代缓存
- 不做“只算部分输出”的假优化
- 允许 shape-specific optimization，但必须是完整合法计算

---

## 三、环境与基础事实

### 1. 评测环境

- GPU: AMD Instinct MI355X (gfx950)
- CPU: AMD EPYC 9575F 64-Core Processor
- Torch: `2.10.0+rocm7.1`
- Triton: `3.6.0`
- Python: `3.12`
- Runtime: ROCm
- 平台：Kubernetes 单容器一次性运行

### 2. aiter 相关事实

- 可注入文件：
  - `aiter/ops/triton/_triton_kernels/quant/quant.py`
  - `aiter/ops/triton/quant/quant.py`
  - `aiter/ops/triton/gemm/basic/gemm_a16wfp4.py`
  - `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py`
- `gemm_a16wfp4_preshuffle` 是当前主力 GEMM 路径
- MI355X 上默认 config 查找经常 miss，日志里会看到：
  - `not found tuned config in CKGEMM or asmGEMM, will use default config!`
- 所以真正有效的做法是：
  - **exact-shape dispatch**
  - **手工传 config**
  - **必要时直接注入 GEMM wrapper / kernel**

### 3. Host-side 噪声的正确理解

- 从 benchmark 角度，当前排名竞争主要仍由 **GPU kernel 时间** 决定
- 虽然 profile/日志中能看到 module build / load / Python 包装开销很大，但 **v57 证明**：单纯绕开 wrapper 不会给 M64/M256 带来实质性 benchmark 收益
- 结论：  
  **host-side 噪声是真问题，但不是当前这道题 1–2µs 级结构突破的主战场**

---

## 四、主线演化（v43 之前简述）

### 阶段 A：安全边界摸清
确认：
- 哪些 API 会触发 500 / another stream
- 顶层 import / Triton launch 的风险
- 源码注入是否可行

### 阶段 B：quant 主线确立
确认：
- 量化是大头
- 分离 quant + shuffle 代价高
- 注入 quant kernel 有明显收益
- bit-manip `_mxfp4_quant_op` 是有效主线

### 阶段 C：preshuffle 主线确立
确认：
- `gemm_a16wfp4_preshuffle` 是强主线
- `skip_reduce=True` 后外部归约可行
- shape-specific config 合法
- 纯 JSON/meta 搜索到 v39-v41 基本接近局部最优

---

## 五、v43-v51：已验证的硬结论

### v43：第一版真正有效的 exact-shape 版本
- 保留 v40 外部数据契约
- 注入 GEMM wrapper + kernel
- 给三组大瓶颈 shape 增加 exact-shape fast path
- exact config 手工传入，不再依赖通用配置查找

代表结论：
- `exact-shape specialization` 是正确方向
- 大 shape 的收益是 **结构性收益**，不是参数偶然

### v44：错误尝试，被明确证伪
- 给 M16 split-K 路径改成原子累加快路径
- 同时微调 M64 / M256 的 exact 调度

结论：
- **M16 的 14-way split-K 上，atomic fp32 accumulate 是错方向**
- **M64/M256 的 exact 调度骨架已经很接近甜点，不该大改**

### v45：当前第一关键突破
- 冻结 v43 在 M64/M256 上的 exact kernel
- 只对 `M16_K7168_N2112` 下刀
- 保留 v43 exact compute
- 新增 **M16 exact reduce kernel**
- 不再走外部 `y_pp.sum(dim=0)`

结论：
- **M16 的瓶颈不在 split-K compute 本身，而在 reduce 方式**
- 一旦换成 exact reduce，M16 从 ~16µs 档掉到 ~11µs 档

### v46-v51：围绕 v45 主线的收敛
- v46：继续精修 M16
- v47b：成为后续新主线基础
- v48/v49：确认 **不再重开新世界线**，主资源继续给 M64/M256 current family
- v50：M256 `b` direct-gather 失败
- v50b：M256 `b_scales` direct-gather 更差
- v51：M256 full-unroll exact kernel 失败

到 v51 为止，已关掉的错路：
- `b` direct-gather
- `b_scales` direct-gather
- M256 full-unroll exact kernel

---

## 六、v52-v59：冲刺阶段的真实收敛

## v52：16×128 资源压力路线的首次正证据
### 做了什么
- M256 改成 `16x128` 子 tile exact 结构
- 不再走 direct-gather，不再删 proven 读法
- 目标是验证“控资源压力”方向，而不是再改 family

### 结果与意义
- M256 从 v50/v50b/v51 的灾难区 **回到 13.9–14.0µs**
- 说明：
  - **资源压力确实是关键变量**
  - 但 **16×128 只是有效支线，不是最终主线**

---

## v53：16×128 + 半展开（失败）
### 做了什么
- 继续 16×128
- 做 3×2 半展开，想压 live range / 寄存器压力

### 结果
- M256 退到 `14.5–14.9µs`

### 结论
- **半展开不是答案**
- 复杂循环组织本身没有带来收益，反而可能恶化资源压力

---

## v54：去掉半展开后的 16×128 exact 验证
### 做了什么
- 保留 16×128 exact
- 去掉 v53 的半展开组织

### 结果
- M256 回到 `13.8–13.9µs`

### 结论
- 16×128 family 有效，但仍然明显慢于主线
- **不能把 16×128 family 升成主线**
- 它的价值主要是帮助确认：  
  **“资源压力”这个判断是对的**

---

## v55：当前主线确立
### 做了什么
- 回到 v49/v47b 那条更快的主家族
- 对 M256 在 **当前最快主家族内部** 做轻量资源压力调优
- 不改读法，不改 tile 家族

### 结果
- `M256_K1536_N3072` 回到 **12.3µs 档**
- `M64_K2048_N7168` 约 **12.7µs**
- 其他 shape 基本不坏

### 结论
- **真正有效的不是换家族，而是在当前最快主家族内控资源压力**
- **v55 成为新 baseline / 新主线**

---

## v55b / v55c / v55d：夜搜候选验证
### 结果概括
- v55b / v55c：能压 M16，但会拖坏全局平衡
- v55d：最接近可用候补，但未稳定打赢 v55

### 结论
- 夜搜挖到的很多正信号，本质是 **M16 单点更好**
- 但当前全局最优由 **M64/M256 的平衡** 决定
- 所以主线继续保持 **v55**

---

## v56：错误迁移——把 M16 的 split/reduce 经验硬套到 M64/M256
### 做了什么
- M64：exact split-K=2 + exact reduce
- M256：exact split-K=3 + exact reduce

### 结果
- M64 退到 **14.8–14.9µs**
- M256 退到 **19.8µs**

### 关键结论
- **M16 的“split-K partial + exact reduce”不能直接迁移到 M64/M256**
- 这次失败不是代码小瑕疵，而是结构假设本身被证伪
- 原因：
  - 对 M64/M256，主动引入 fp32 partial buffer + reduce，相当于人为制造了大量额外全局流量
  - 而这两组当前最强路径本来就是 **one-pass exact path**

> 需要修正的方法论：  
> **reduction 只有在 split-K 本身已经必要/已证实有利时，才值得升格成主战场。**

---

## v57：direct exact launch 绕开 wrapper（高质量负结果）
### 做了什么
- 保持 v55 current family
- 对 M64/M256 在 `custom_kernel()` 里直接 exact launch
- 绕开 `gemm_a16wfp4_preshuffle` 的通用 Python wrapper / dispatch

### 结果
- M64 基本还是 **13.0µs**
- M256 基本还是 **12.4–12.5µs**

### 关键结论
- **wrapper / Python dispatch 不是 M64/M256 的主瓶颈**
- 这条分支可以关闭：
  - 继续砍 wrapper
  - 继续绕开 exact dispatch 外层包装
- 以后真正值得打的热点，只剩 **kernel 内部**

---

## v58：M64 family switch（失败，但信息量很大）
### 做了什么
- 只对 M64 换 family
- 从 current fused family 切到：
  - `dynamic_mxfp4_quant(A)`
  - `e8m0_shuffle(A_scale)`
  - `aiter.gemm_a4w4(..., bpreshuffle=True)`

### 结果
- `M64_K2048_N7168` 从 ~12.7µs 直接炸到 **24.6–24.8µs**
- M256 基本没动，仍然约 **12.3–12.5µs**

### 关键结论
- **M64 的结构性突破不能靠“拆掉当前 fused 主家族，换成分离 quant+gemm family”来拿**
- 这不是小退步，而是成倍退化
- 本质原因：
  - 把当前主线最值钱的 fused/injected 优势打散了
  - 又重新引入了分离 quant / shuffle 的代价

所以 v58 关闭的不是“跳出局部最优”这个大方向，而是更具体的一条：

> **M64 的 family switch，不能走“分离 quant + aiter.gemm_a4w4(bpreshuffle)”这条路。**

---

## v59：current family 内部热点静态化（已制作，待复测）
### 做了什么
- 只动 **M64 exact kernel**
- 保持 current fused family
- 保持 one-pass
- 保持 B 的 proven block-load 路径
- **只把 `b_scales` 的 layout 解释，从 load 后 `reshape/permute` 改成 exact-direct 指针布局**
- loop 改成 `tl.static_range(0, 8)`，但不做 full-unroll 爆炸
- 其他 5 组完全继承 v55

### 当前假设
v59 只验证一个非常具体的问题：

> **M64 current family 内部，真正值得打的一处热点，是每轮 `b_scales` 的局部 layout 解释成本。**

### 这版的价值
- 如果赢了：说明 M64 的结构突破口确实在 **kernel 内部 runtime interpretation**
- 如果输了：也很有价值，因为说明：
  - M64 的主热点不在 `b_scales` 这层解释
  - 更可能在：
    - B operand 本体
    - dot 前后的 live range / 调度
    - 其他 kernel 内部 generic logic

**当前状态：待你重新 benchmark 验证，不应预先下结论。**

---

## 七、截至 v59，已经明确关掉的错路

### M16 相关
- split-K + atomic fp32 accumulate

### M256 相关
- `b` direct-gather
- `b_scales` direct-gather
- full-unroll exact kernel
- half-unroll / 复杂循环组织
- 16×128 family 升主线
- split-K partial + exact reduce
- wrapper 直连 exact launch 作为主要突破点

### M64 相关
- split-K partial + exact reduce
- direct family switch 到 `dynamic_mxfp4_quant + gemm_a4w4`
- wrapper 直连 exact launch 作为主要突破点

---

## 八、当前最优认识（截至 v59）

## 1. 主线已经固定
未来版本都基于：

- quant 注入
- preshuffle 主线
- exact-shape dispatch
- M16 exact compute + exact reduce
- M64/M256 current fused family one-pass exact path
- 资源压力只允许在 current fastest family 内部处理

---

## 2. 三组关键 shape 的现状

### A. `M16_K7168_N2112`
- 已经从 ~16µs 档被打到 ~10µs 档
- 仍可小抠，但已经不是全局第一主矛盾
- 不能为了再省 0.1–0.2µs 把 M64/M256 打坏

### B. `M64_K2048_N7168`
- 当前主线约 **12.7µs**
- v56/v57/v58 分别关掉了：
  - split+reduce
  - wrapper 路线
  - 分离 quant+gemm family switch
- 现在最值得押的是：
  - **current family 内部 kernel 热点静态化**
  - 也就是 v59 / 后续 v60 这类方向

### C. `M256_K1536_N3072`
- 当前主线约 **12.3µs**
- 16×128 证明了“资源压力”是对的，但不是最终主线
- v56/v57 证明：
  - 不是 reduce
  - 也不是 wrapper
- 现在最该继续打的是：
  - `b / b_scales` 周围的 inner-loop 解释成本
  - pointer/index 重复计算
  - live range / 寄存器压力

---

## 3. K=512 那三组的位置
- 它们已经不再是主战场
- 但在冲刺到 8.2 的最后阶段，每组再省 `0.1–0.2µs` 仍然值钱
- 所以后续版本里允许“顺手优化”，但不能喧宾夺主

---

## 九、从外部资料中真正吸收到的方法论

### 1. report4 / 其他报告真正适合迁移的，不是 distributed 技巧本身
不该机械照搬：
- IPC
- symmetric heap
- global barrier
- XCD remap
- 分布式通信-计算重叠

当前这道题是 **单卡 benchmark**，这些都不是主战场。

### 2. 真正值得吸收的是三条原则
#### A. 把最热的通用逻辑变成 exact-path
这与 v43 之后的 exact-shape 路线一致。

#### B. 只有 split 已经必要时，reduction 才值得单独升格优化
M16 属于这种情况，所以 v45 成功；M64/M256 目前不属于，所以 v56 失败。

#### C. 把运行时解释成本变成编译期常量
这点最值得继续迁移到 M64/M256：
- pid mapping
- 边界/掩码逻辑
- pointer arithmetic
- layout interpretation

---

## 十、当前的后续规划（基于 v59 之后的正确顺序）

### 短期
1. **先重新 benchmark v59**
2. 根据 v59 结果判断：
   - 若 M64 下去：继续 current-family kernel hotspot 路线
   - 若 M64 不动或更差：说明 `b_scales` 不是主热点，下一刀应转向 M64 其他 kernel 内部 generic logic

### 中期
#### M64
优先顺序：
1. current family 内核热点静态化
2. 只在证据明确时再考虑其他 kernel-internal surgery

#### M256
优先顺序：
1. 保持 current family
2. 继续打 inner-loop 热点：
   - `b / b_scales` 局部 layout 解释
   - pointer/index 重复计算
   - live range / register pressure
3. 不再走：
   - split/reduce
   - wrapper
   - 16×128 主线化
   - 分离 family switch

---

## 十一、给未来自己的操作守则

1. **不要因为一次负结果就否定“结构化尝试”本身**
   - 要先分清：是方法错，还是实现层次不对

2. **高质量失败必须能关掉一整条分支**
   - v56 关掉 split+reduce
   - v57 关掉 wrapper
   - v58 关掉分离 quant+gemm family switch

3. **每个新版本最好只验证一个核心假设**
   - v59 就是很好的形式：只打 M64，只打一处热点

4. **主线优先级永远高于局部漂亮数据**
   - M16 再省一点很好，但不能拿 M64/M256 去换

5. **不做 exploit，不走灰线**
   - 不缓存
   - 不跳算
   - 不做非 production-worthy 技巧

---

## 十二、当前最短总结

> **v55 仍然是当前主线。**
>  
> **v56、v57、v58 都不是白失败，它们分别帮我们高质量关掉了：split/reduce、wrapper、分离 family switch 这三条看起来合理但实际上错误的结构分支。**
>  
> **v59 是当前最像正确下一刀的尝试：不再从 kernel 外面打，而是进入 current fused family 的 kernel 本体，验证 M64 的 `b_scales` 局部 layout 解释是否是真热点。**
