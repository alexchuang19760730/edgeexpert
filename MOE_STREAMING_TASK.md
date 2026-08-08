# MoE Expert Streaming 优化任务 — Prime Agent 初始提示

## 你的身份与工作环境

你是 Prime Agent，在一个**隔离的工作树**中工作：
```
/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/
```
这是 TurboFieldfare（Swift/Metal 的 Gemma 4 26B-A4B MoE 推理引擎）的完整源码副本，**不含编译产物**。

**重要环境事实：**
- 编译命令：`export PATH="/opt/homebrew/opt/swift/bin:$PATH" && swift build -c release --target TurboFieldfareCLI`
- 模型权重**不在此工作树**（13GB，避免复制），运行时通过绝对路径引用：
  - r4（4-bit）：`/Users/alexchuang/Documents/flashkv0516/models/gemma4.gturbo`
  - r3（3-bit）：`/Users/alexchuang/Documents/flashkv0516/models/gemma4-r3.gturbo`
  - r2（2-bit）：`/Users/alexchuang/Documents/flashkv0516/models/gemma4-r2.gturbo`
- hot-pool profiles 在 `/Users/alexchuang/Documents/flashkv0516/models/profiles/`（top32/48/64/80_code.json）
- 基准 prompt：`/tmp/msg_code.json`（若被系统清理，内容为：`[{"role":"user","content":"Write a Python function that computes the n-th Fibonacci number using memoization. Include a short docstring."}]`）
- 机器：Mac M4 / 16GB 统一内存。**16GB 是关键约束**——专家权重无法全驻留，必须流式加载。

## 项目背景（必读文档）

先读这些文件再动手：
1. **`docs/TURBOFIELDFARE_MMAP_EXPERT_IO_PLAN_2026-08-07.md`（最重要，45KB）**——完整记录了 2026-08-07 的全部实验、结论与代码状态
2. `Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift` —— 核心文件（expert 流式加载）
3. `Sources/TurboFieldfare/Infrastructure/Streaming/ExpertStreamer.swift` —— streamer 抽象
4. `Sources/TurboFieldfare/Kernels/MoE/MoE.swift` —— MoE 调用路径
5. `bin/run_prod.sh` —— 生产启动配置（已固化所有已验证优化）
6. `bin/bench_ab.sh` —— A/B 对比工具（new|base|ab 三模式）

## 已知结论（2026-08-07 已定案，不要重走弯路）

**这些方向已经被实验证明是死路，不要再尝试：**
- ❌ **mmap 专家 IO（`TURBO_FIELDFARE_EXPERT_MMAP=1`）**：Metal 对 file-backed buffer 的驻留税（~85µs/buffer/CB）> IO 节省，16GB 机器上结构性输给 pread；且有 GPU 页回收卡死风险。文档 §7 已完整验证。
- ❌ **多 MTLCommandQueue 并发**：decode 依赖链是严格串行链，第二 queue 无独立工作可跑。文档 §10.3。
- ❌ **MTP 投机解码靠常驻转正**：MTP 成本 = batched verify 的 GPU 前向计算，与专家 IO 解耦，100% 常驻也不会转正。文档 §8.5。
- ❌ **CB 融合（B1）**：已测回归。

**有效且已落地的优化（生产默认开启，见 run_prod.sh）：**
- ✅ hot pool（pinned 高频专家，code +25%）：`TURBO_FIELDFARE_HOT_POOL=1` + profile（默认 top-48/64）
- ✅ 专家缓存槽 64-96（hit 95%+）
- ✅ 2/3-bit 量化（r2/r3，bytes 减半）
- ✅ MPP tensor-ops decode attention（`TURBO_FIELDFARE_ATTN_TENSOROPS=1`，CPU 侧 cb 开销下降）
- ✅ `--trust-receipt`（TTFT -3.9s）

## 当前瓶颈（2026-08-07 实测，r4 生产配置）

```
[gpu] attn=2.03s(36%) routedMoE=1.70s sharedFFN=0.79s phase1Hit=0.32s head=0.83s busy=5.67s ofWall=44%
[sync] cbs=12760 cb1Wait=5.60s sched=7.32s overhead=0.87s(68us/cb) ofWall=45%
[stage] cb1=0.16s(1%) io=2.21s(16%) gpuWait=10.06s(75%) wall=13.49s
decode = 9.23s (128 tok, 72ms/step)
```

**当前 decode 约 15-17 tok/s（r4 pool64）**。用户目标是突破 20 tok/s。

## 任务目标（按优先级）

### P0：验证并量化当前生产配置的真实 tok/s
- 用 `bin/run_prod.sh`（r4 生产配置）跑基准，确认基线数字
- 用 `bin/bench_ab.sh` 或手工交错 A/B 对比（**必须交错，防页缓存漂移**）

### P1：探索 20 tok/s 路径（按文档 §10.4 ROI 排序）
1. **更大的 hot pool / 更聪明的 profile**：top-64 vs top-48 vs top-80 的覆盖率-内存权衡（文档说 pool64 = 97.1% 覆盖 + sync preload，decode +32%）
2. **CPU 侧调度**：文档 §10 显示 gpuWait=75%（GPU 等 CPU 喂活）、sched=7.32s。已知多 queue 无效、encode 移后台只有 1%，**但** main-thread readback→plan→fetch-await→routed-encode 的串行链（~0.6ms/layer）是否还能压缩？例如：routed expert 的 plan/fetch 提前到前一层 GPU 执行期间（流水线化），而不是等 readback 完。
3. **r2 量化**（如果质量可接受）：文档说 r2 decode +68%。

### P2：质量与稳定性
- 任何改动必须保持输出与改动前**逐字节一致**（文档 §9.6 验证方法）
- 每项优化用交错 A/B 验证，记录到 docs/ 新文档
- 不要在未验证的情况下改动生产默认值

## 边界与安全

- **这是隔离工作树**，可以自由改代码、编译、跑基准。**不要动工作树外的任何文件**（模型目录、原仓库、profile）。
- 模型引用用绝对路径（上面给的三条），不要复制模型。
- 磁盘只有 ~32GB 空闲：不要创建大文件；编译用 `swift build`（产物在 .build/，可接受），但**不要**跑 `swift package clean` 后重建全量依赖超过必要次数（每次 ~5-8 分钟）。
- 你的每轮改动：先编译 → 跑输出一致性验证 → 交错 A/B → 汇报。不要一次改完所有东西再验证。
- 最终产出：一份 `docs/OPTIMIZATION_RESULTS_<date>.md`，记录：每项尝试、A/B 数据、结论、是否保留。

## 汇报格式

每完成一个阶段，用简洁中文汇报：
```
[阶段] 目标
[结果] 关键数字（tok/s、TTFT、改动前后）
[结论] 有效/无效/保留建议
[下一步] 建议
```

现在开始：先读文档 §7-§12 和 run_prod.sh，然后跑 P0 基线。
