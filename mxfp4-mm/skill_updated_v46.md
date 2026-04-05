# AMD MI355X MXFP4-MM 技能库（更新版，至 v46）

> 更新日期：2026-04-04  
> 目标：在 **合法、可复现、可提交** 的前提下，把当前 **v45 benchmark geomean ≈ 8.90µs** 继续压向 **8.2µs** 附近。  
> 当前主线：**quant 注入 + preshuffle 主线 + exact-shape GEMM specialization + M16 exact reduce**。

---

## 一、当前一句话结论

这条路已经被证实是对的：

- **v39-v41 纯参数搜索** 基本到头，长期停在 **10.5µs 左右**
- **v43** 证明：**exact-shape kernel specialization** 是正确方向
- **v44** 证明：**M16 的 split-K 原子累加** 是错误方向
- **v45** 证明：**M16 exact compute + exact reduce** 是当前最大突破口，直接把 `M16_K7168_N2112` 从 **~16.1µs** 砍到 **~10.8µs**
- 后续版本不再重构主线，而是围绕 **v45** 做 disciplined tuning

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
- `torch.utils.cpp_extension.load_inline()` **现在看是可用的**，但目前没有证明它比 Triton 注入主线更快

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

---

## 四、到目前为止的主线演化

## 阶段 A：安全边界摸清
早期版本主要是在确认：

- 哪些 API 会触发 500 / another stream
- 顶层 import / Triton launch 的风险
- 源码注入是否可行

这一阶段的价值不是性能，而是把“可提交的武器库”找出来。

---

## 阶段 B：quant 主线确立
后面逐步确认：

- 量化是大头
- 分离 quant + shuffle 代价高
- 注入 quant kernel 有明显收益
- bit-manip 版本 `_mxfp4_quant_op` 是有效主线

这一步把 baseline 的二十多微秒压下来了。

---

## 阶段 C：preshuffle 主线确立
再往后确认：

- `gemm_a16wfp4_preshuffle` 是强主线
- `skip_reduce=True` 后外部归约可行
- shape-specific config 合法
- 继续靠 JSON 配置搜索还能提升，但幅度有限

v39-v41 基本属于这一阶段的末端：  
**方向没错，但纯参数搜索已经接近局部最优。**

---

## 阶段 D：exact-shape 体系建立（v43 起）
真正的结构性突破从这里开始：

- 不再主要依赖 `_get_config()`
- 开始按官方 6 个 benchmark shape 做 **exact-shape dispatch**
- 对大 shape 写 **exact fast path**
- 外部数据契约保持不变，不再做 Python/Torch 侧昂贵重排

这是从 **10.5µs 档** 往 **9µs 档** 迈的关键一步。

---

## 五、v42-v46 的真实经验总结

## v42：失败，但价值很大
### 做了什么
- 试图把 `w_scales` 改成新的 direct layout
- 在 Python/Torch 热路径里做了 scale 预排布

### 结果
- 6 个 shape **全线退化**
- 说明问题不是单个参数，而是公共热路径变重

### 核心结论
**不能把任何会 materialize 新 tensor 的重排放进热路径。**

特别是：
- `permute(...).contiguous()`
- reshape 后实际发生 copy 的路径

这些都会被 benchmark 计时，代价会吃满所有 shape。

### 固化规则
以后热路径里只允许：
- `view`
- `reshape`
- 简单 slicing
- 已知零成本的 layout 变换

---

## v43：第一版真正有效的 exact-shape 版本
### 做了什么
- 保留 v40 外部数据契约
- 注入 GEMM wrapper + kernel
- 给三组大瓶颈 shape 增加 exact-shape fast path
- exact config 手工传入，不再依赖通用配置查找

### 代表结果
代表 benchmark 大约到：
- `M4_K512_N2880` ≈ `6.18µs`
- `M16_K7168_N2112` ≈ `16.1µs`
- `M32_K512_N4096` ≈ `6.96µs`
- `M32_K512_N2880` ≈ `6.96µs`
- `M64_K2048_N7168` ≈ `12.8-12.9µs`
- `M256_K1536_N3072` ≈ `12.4-12.5µs`

### 价值
- 证明了 **exact-shape specialization 是对的**
- 证明了大 shape 的收益不是“参数偶然”，而是结构性收益
- 几何均值大约到 **9.58µs**

### 仍然没解决的问题
- `M16_K7168_N2112` 仍然过慢
- 主瓶颈从“很多地方都慢”收缩成了“一个最难啃的大 shape 明显偏慢”

---

## v44：错误尝试，被明确证伪
### 做了什么
- 给 M16 split-K 路径改成原子累加快路径
- 同时微调 M64 / M256 的 exact 调度

### 结果
- `M16` 从 ~16.1µs **反弹到 ~20.5-20.9µs**
- `M64` 从 ~12.8µs 回退到 ~13.4-13.7µs
- `M256` 从 ~12.4µs 回退到 ~14.0µs

### 关键结论
1. **M16 的 14-way split-K 上，atomic fp32 accumulate 是错方向**
2. **M64/M256 的 exact 调度骨架已经很接近甜点，不该大改**

### 固化规则
- 不再对 M64 / M256 做“大骨架改写”
- M16 继续 split-K，但必须换一种 reduce 方式

---

## v45：当前最关键的突破版本
### 做了什么
- 冻结 v43 在 M64/M256 上的 exact kernel
- 只对 `M16_K7168_N2112` 下刀
- 保留 v43 exact compute
- 新增 **M16 exact reduce kernel**
- 不再走外部 `y_pp.sum(dim=0)`

### 代表结果（已多次验证）
- `M4_K512_N2880` ≈ `6.20µs`
- `M16_K7168_N2112` ≈ `10.8-11.0µs`
- `M32_K512_N4096` ≈ `6.83-6.90µs`
- `M32_K512_N2880` ≈ `6.83-6.88µs`
- `M64_K2048_N7168` ≈ `12.8-12.9µs`
- `M256_K1536_N3072` ≈ `12.3-12.6µs`

### 关键意义
这是当前最重要的结论：

**M16 的瓶颈不在 split-K compute 本身，而在 reduce 方式。**

一旦把它换成 exact reduce，M16 直接从 ~16µs 档掉到 ~11µs 档。  
这一步把整体 benchmark geomean 压到 **约 8.90µs**。

### 当前地位
**v45 是当前新的主基线。**

---

## v46：基于 v45 的小步精修（已做，待更多结果验证）
### 做了什么
- 只继续精修 M16
- 把 exact reduce 从 `N=64` 尝试提到 `N=128`
- 给 M16 的 `b` / `b_scales` 读取加 `.cg`

### 目的
- 继续榨 M16 最后那一点空间
- 不破坏 v45 已经拿到的全局收益

### 当前结论
- 方向是 **小步、可解释、低风险** 的
- 是否优于 v45，需要更多 benchmark 结果确认
- 即使 v46 失败，也不影响 v45 作为当前主基线的地位

---

## 六、当前最佳认识（截至 v46）

## 1. 主线已经固定
未来版本都基于：

- quant 注入
- preshuffle 主线
- exact-shape dispatch
- M64/M256 exact fast path
- M16 exact compute + exact reduce

这条主线 **不再推翻重来**。

---

## 2. 三组大 shape 的现状

### A. `M16_K7168_N2112`
- 曾经是最大瓶颈
- v45 已经把它砍到 ~10.8µs
- 现在仍然是最值得继续抠的单点
- 但优化重点已经从“换方向”变成“微调 reduce / split / load”

### B. `M64_K2048_N7168`
- 已从 v40 的 ~13.3µs 档进入 ~12.8µs
- 说明 exact kernel + `waves=4` 有价值
- 但不要再大改 exact 调度骨架
- 后续只做 meta 参数微调

### C. `M256_K1536_N3072`
- 已从 ~16µs 档直接压到 ~12.3-12.6µs
- 这是 v43 exact-shape 体系最成功的案例之一
- 后续只允许小范围扫：
  - `num_warps`
  - `GROUP_SIZE_M`
  - `cache_modifier`
- 不要再重写调度骨架

---

## 3. K=512 那三组的位置
- 它们已经不再是主战场
- 但在冲刺到 8.2 的最后阶段，每组再省 `0.1-0.2µs` 仍然值钱
- 所以后续版本里允许“顺手优化”，但不能喧宾夺主

---

## 七、当前最优配置认识

下面这些结论已经有较强证据支持：

### M16_K7168_N2112
- `BLOCK_SIZE_M=16`
- `BLOCK_SIZE_N=128`
- `BLOCK_SIZE_K=256`
- `num_warps=8`
- `num_stages=2`
- `waves_per_eu=2`
- `NUM_KSPLIT=14`
- `cache_modifier=None` 或 `.cg` 都值得继续观察，但 exact reduce 才是决定性因素

### M64_K2048_N7168
- 方向上 `BM=16 / BN=128 / BK=256`
- `waves_per_eu=4` 在当前 exact 路线下值得保留
- 不再大改调度，只做参数级细调

### M256_K1536_N3072
- `BM=16 / BN=256 / BK=256` 是当前强方向
- `waves_per_eu=2`
- `num_warps=8` 是当前默认优选
- exact kernel 比旧的通用参数路线明显更强

### K=512 三组
- `M4_K512_N2880`：`BLOCK_SIZE_N=32` 有价值
- `M32_K512_N4096`：`waves_per_eu=1 + num_warps=4` 有价值
- `M32_K512_N2880`：`cache_modifier=None` 曾表现出价值

---

## 八、已经被证伪的方向

这些内容以后不再浪费提交：

1. **Python/Torch 热路径里重排 `w_scales` / `w`**
   - 特别是 `permute + contiguous`
   - v42 已证伪

2. **M16 的 split-K atomic fp32 accumulate**
   - v44 已证伪

3. **重写 M64/M256 的 exact 调度骨架**
   - v44 已证伪

4. **回到纯 JSON / config 搜索作为主线**
   - v39-v41 已说明边际收益太低

5. **任何 exploit / cache / skip compute 方案**
   - 合规风险高
   - 后续 leaderboard 也在清理

---

## 九、未来规划路径（冲刺版）

## 目标
从当前 **v45 benchmark geomean ≈ 8.90µs** 继续向 **8.2µs** 推进。

要做到这点，预计需要：

- `M16` 再省 `0.3-0.8µs`
- `M64` 再省 `0.8-1.2µs`
- `M256` 再省 `0.8-1.3µs`
- K=512 三组合计再省 `0.2-0.4µs`

---

## 路线 1：v46/v47 —— 继续精修 M16
### 目的
把 `M16` 从 ~10.8µs 压到 **10.0µs 甚至以下**

### 优先实验
1. `NUM_KSPLIT = 10 / 12 / 14`
2. exact reduce `BLOCK_SIZE_N = 32 / 64 / 128`
3. `cache_modifier = None / .cg`
4. 只优化 M16 的 B/B-scale load 路径

### 原则
- **只改 M16**
- M64/M256 冻结
- 绝不把 v45 的收益打散

---

## 路线 2：v47/v48 —— 微调 M64 / M256
### 目的
把两组从 ~12.8 / ~12.3 再往 **11µs 左右** 推

### M64 值得扫的项
- `waves_per_eu = 2 / 4`
- `BLOCK_SIZE_N = 128 / 256`
- `cache_modifier = .cg / None`

### M256 值得扫的项
- `num_warps = 4 / 8`
- `GROUP_SIZE_M = 4 / 8`
- `cache_modifier = .cg / None`

### 禁止项
- 不改 exact kernel 骨架
- 不改外部数据契约
- 不引入新布局

---

## 路线 3：最后阶段再收 K=512 三组
### 目的
吃掉最后的小边际收益

### 可做
- `M4`：BN / waves / warps 小扫
- `M32_N4096`：继续围绕 `waves=1, warps=4`
- `M32_N2880`：继续围绕 `cache_modifier`

### 但优先级最低
K=512 不是冲榜主战场。

---

## 路线 4：高风险支线（只在主线收敛后考虑）
1. 更安全地借用 ASM tuned kernel
2. 更激进的 GEMM kernel 内部重排优化
3. HIP C++ / inline ASM 只作为支线验证，不进主线

注意：  
这些是“备选”，不是当前冲刺主线。  
当前最有效的路径仍然是 **v45 主线 + disciplined tuning**。

---

## 十、提交与实验纪律

### 每个新版本必须回答三个问题
1. 这版只改了什么？
2. 它针对的是哪一个已知瓶颈？
3. 如果失败，能否明确说明“哪条路被证伪了”？

### 提交纪律
- 一次只改一个结构点
- 结构点有效后，再做参数小扫
- 不允许“多个大方向同时上”
- 所有 benchmark 回来后必须记录：
  - 6 shape 时间
  - geomean
  - 改了什么
  - 是否保留

### 当前基线
- **提交主基线：v45**
- **开发分支：v46 及以后**
- 如果新版本没有明确超过 v45，就直接回到 v45

---

## 十一、当前推荐工作流

### 1. 主线代码
始终从 **v45** 分支出来。

### 2. 每个新版本的模式
- `v46a`: 只改 M16 exact reduce
- `v46b`: 只改 M16 split 参数
- `v47a`: 只改 M64 小参数
- `v47b`: 只改 M256 小参数
- `v48`: 再收 K=512

### 3. 判断标准
不是“感觉快了”，而是：
- 是否压低 benchmark geomean
- 是否压低大 shape
- 是否没有破坏其他收益点

---

## 十二、当前最重要的行动建议

如果现在继续往前做，最优先顺序应当是：

1. **先把 M16 再抠一层**
2. **然后压 M64**
3. **再压 M256**
4. **最后收 K=512 三组**

因为当前已经不是“有没有方向”的问题，而是：
**如何把 v45 这条正确路线继续有纪律地榨干。**

---

## 十三、当前状态快照（建议作为后续版本对照基线）

### 当前最可信 benchmark 基线（v45）
- M4_K512_N2880 ≈ 6.20
- M16_K7168_N2112 ≈ 10.8
- M32_K512_N4096 ≈ 6.83
- M32_K512_N2880 ≈ 6.88
- M64_K2048_N7168 ≈ 12.8
- M256_K1536_N3072 ≈ 12.3

### 对应 benchmark geomean
- **≈ 8.90µs**

### 当前战略判断
- 已从“探索主线”进入“冲刺榨干阶段”
- 主线正确，不再重构
- 目标是通过连续 2-4 个 disciplined 版本，逼近 **8.2µs**

---

## 十四、最后的行动原则

以后每一版都遵守这三条：

1. **不为了“更激进”而打散已知收益**
2. **不为了“看起来更优雅”而把重排塞回热路径**
3. **每次失败都要沉淀成下一条硬规则**

这份文档就是为了把这些规则固定下来，避免重复踩坑。
