# TurboFieldfare Routed-Expert IO 改造方案：pread → mmap + 页缓存预读

日期：2026-08-07（磁盘清理后，42Gi 空闲）
目标：把 routed expert 权重从「pread 拷贝进私有 slot」改为「mmap + 页缓存直读」，让多个 token（尤其 MTP batched verify）共享同一批已缓存专家页。

---

## 1. 现状（当前 pread 数据面）

`PreadExpertStreamer.swift`（每层一个 streamer，`ExpertStreamingMode.pread(slotCount:)`，默认 16，生产 64）：

1. init：`open(layer_file, O_RDONLY)` + `posix_memalign` 分配 `slotCount` 个页对齐私有缓冲（`slotPointers`），每个 `device.makeBuffer(bytesNoCopy:, .storageModeShared)` 包成 MTLBuffer。
2. miss 路径：`loadExpert` → `readFull` → **循环 `pread(fd, slotPtr, stride, streamOffset+expertOffset)`**，把整块 expert（r4 3.2MB / r3 2.5MB）从页缓存**拷贝进私有 slot**。
3. hit 判定：slot 标签表 `slotExpert[slot] == expert`（有限槽的启发式），LRU 换出。
4. 预取：`fcntl(fd, F_RDADVISE)`（`RDAdvice.swift`）+ lookahead 预读进 slot。
5. 对比：**resident 权重早已是 mmap 模式**（`ResidentBuffer.swift`：`mmap(PROT_READ, MAP_PRIVATE)` → `posix_madvise(POSIX_MADV_RANDOM)` → `makeBuffer(bytesNoCopy:, .storageModeShared, deallocator: munmap)`）。routed 是唯一还在 pread 拷贝的数据面。

### 关键数字（本次会话实测）

| 项 | 值 |
|---|---|
| r4 expert stride | 3,358,720 B = **205 × 16,384（页对齐 ✓）** |
| r3 expert stride | 2,605,056 B = **159 × 16,384（页对齐 ✓）** |
| 层 × 专家 | 30 × 128，专家权重合计 ≈ 12.9GB |
| slot 内存占用 | 30 层 × 64 槽 × 3.2MB ≈ **6.3GB**（128 槽 = 12.6GB，压垮 16GB 机） |
| slots=64（干净机，r4） | hit 95.7%，pread 9.9GiB/128tok，tok/s ≈ 15.9 |
| slots=16 | hit 62.7%，pread 18GiB/64tok，decode 7-9 tok/s |

---

## 2. 改造方案：mmap + mincore 规划 + MADV_WILLNEED 预读

### 2.1 数据面（核心，改动最小）

在 `PreadExpertStreamer` 内新增 `useMmap` 模式（或独立 `MmapExpertStreamer`，公共 API 不变）：

- **init**：`mmap(nil, streamSize, PROT_READ, MAP_PRIVATE, fd, 0)` 整层文件一次映射（`streamOffset=0`、每层独立文件，天然成立）。
- **专家缓冲**：`device.makeBuffer(bytesNoCopy: mappedBase + expertOffset, length: stride, .storageModeShared)` —— 每个专家一个视图，**3840 个 buffer 全部指向同一块 mmap，零额外内存**；首次访问时惰性创建并缓存（字典），**永不驱逐**。
- **loadExpert**：字典查表（命中即返回 buffer），无 pread、无拷贝。GPU 直读页缓存页（与 resident 权重同一已证明模式）。
- **hit/miss 规划**：把 slot 标签表换成 **`mincore(mapped + offset, stride, vec)` 真页驻留判定**——「hit」= 页确实在内存，比「曾经装进某槽」精确得多（槽被 LRU 换出但页仍驻留的专家，今天会白白 pread 重读，mmap 下是真 hit）。
- **预取**：保留 `F_RDADVISE`，并新增 **`madvise(addr, len, MADV_WILLNEED)`** 对 lookahead 预测的下一层专家窗口做精确预读（比 fd 级 advice 更精准）。
- **deinit**：先释放所有 MTLBuffer，再 `munmap`。

### 2.2 接线

- `ExpertStreamingMode` 增加 `mmap`（或 `.pread(slotCount:useMmap:)`）；`Model.swift` 按模式选 streamer。
- 新 env 旋钮：`TURBO_FIELDFARE_EXPERT_MMAP=1`（默认 off，回退 pread）。
- telemetry：expert-cache 命中率改为 mincore 口径，新增 `[expert-io]` 的 mmap fault 统计。

### 2.3 为什么内存不再随 slot 增长

slot 缓冲从「每个槽私有分配」变成「全部视图同一块 mmap」——**slot 数概念对内存完全无成本**。128+ 槽、甚至「所有 3840 个专家 buffer 全活」都只占 mmap 元数据 + 3840 个轻量 MTLBuffer 对象（合计 < 几 MB）。页是否驻留由 OS 页缓存统一裁决（42Gi 空闲 vs 12.9GB 专家，热集可整驻）。

---

## 3. 预期收益（量化）

### 3.1 内存：−6.3GB（64 槽）或解锁 128+ 槽

30×64×3.2MB ≈ **6.3GB 私有 slot 分配归还系统**（128 槽 12.6GB 同理解锁）。16GB 机上这是巨大呼吸空间，且「128 槽压垮机器」的旧结论作废。

### 3.2 decode / TTFT 量化预估（2026-08-07 干净机 telemetry 推算）

基线（slots=64，B-tree prompt，128 tok，本会话实测）：

| 模型 | decode | tok/s | readWall 占 decode | GPU busy（floor） | GPU ceiling |
|---|---|---|---|---|---|
| r3 | 8.96s | 14.3 | 4.52s（**50%**） | 5.73s | ~25 tok/s |
| r4 | 8.05s* | 15.9* | ~63%（噪声run 8.5s/13.5s） | ~7.7s | ~19 tok/s |

（*r4 取 B1 交错中位；telemetry 单跑噪声大。）

**IO 为什么占一半：命中率 94-97% 下 readWall 仍巨大**——(a) lookahead 预取每轮 pread 拷贝 7-8GiB 进 slot，命中 65% 即 ~35% 白拷；(b) 被 LRU 换出但页仍在缓存的专家被白白重读重拷（slot 口径 hit 低估页驻留）。

**mmap 后预估（页缓存 42Gi 承载 12.9GB 专家，热集整驻）：**

| 项 | 改造前（r3） | 改造后预估（r3） | 改造后预估（r4） |
|---|---|---|---|
| decode tok/s | 13-14 | **19-22**（+40~60%） | **17-19**（接近 GPU ceiling ~19） |
| TTFT | 2.6-2.9s | **2.0-2.5s**（−15~25%） | **2.6-3.1s**（−10~20%） |

推算逻辑：
- decode：`readWall → 首触 fault 为主（预取变页预热、驻留重读变真 hit、拷贝与 syscall 消除）`，IO 占比 50%→~15-20%；decode 时间 → GPU busy + head + 残余 sched ≈ r3 5.7-6.2s / r4 6.8-7.5s。
- 上限即 GPU ceiling（r4 ~19 / r3 ~25 tok/s，按 147 token 当量 GPU busy 推算）——mmap 之后 decode 将贴近此值，再往上需 GPU 侧优化（head/attention）。
- TTFT：prefill 专家读同样去拷贝/去预取白拷，但首触磁盘读仍存在，故增益小于 decode。



- 现状瓶颈（slots=64 干净机，r4 ~15.9 / r3 ~13.0）：每 miss 仍是一次 pread 系统调用 + 3.2MB 页缓存→私有页拷贝 + LRU 驱逐重读。
- mmap 后：miss 变成一次页 fault（首次触碰该专家才发生）；热集专家重访问全部页 hit、零拷贝。命中率从 95.7%（槽口径）升到 **~99%+（页驻留口径）**。
- 残余成本只剩「首次触碰的不同专家必须从盘读一次」——这是任何方案都免不了的，但 42Gi 页缓存 + 12.9GB 专家使其在预热后趋近零。
- 对应「IO 占 decode 比例 56.9%→42.7%（4→2-bit）」的既有下降曲线，mmap 把该比例推向 <10%。

### 3.3 MTP batched verify：从 −10~15% 转向持平或转正

- 今天 MTP 失败机制：verify 批内 N 行 expert 并集（≤8N 个），slots=32 时槽颠簸 → 每行重读重拷。mmap + mincore 下并集专家页**批内共享、批间驻留**，batched verify 不再重复付 pread 拷贝。
- 接受率已健康（r4 68% / r3 82%），若 verify 的 IO 边际成本归零，投机解码在 r3 上转正的现实性大增。

### 3.4 规划精度与跨进程共享

- mincore 口径让 prefetch planner 只对真缺失页发 F_RDADVISE/MADV_WILLNEED（今天槽口径会误发）。
- 页缓存全局共享：Server/worker 多进程共用热专家页。

---

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Metal `bytesNoCopy` 包 mmap 内存 | **同一代码库 resident 权重已生产使用**（ResidentBuffer）；窗口天然 16K 页对齐（stride=205/159 页） |
| GPU 访问被换出的页 | MAP_PRIVATE 页驱逐后 GPU fault 经统一内存管理器重新调入（与 resident 路径今天的行为相同）；MADV_WILLNEED 预读缓解抖动 |
| buffer 生命周期 vs munmap | deinit 先 release 全部 buffer 再 munmap（ResidentBuffer 已示范 deallocator 捕获模式） |
| 热路径回归 | 新模式默认 off（env 开启），A/B 与 pread 严格交错对比；公共 API 不变，可随时回退 |
| mincore 粒度 | macOS 支持；stride 对齐页边界，判定无边界瑕疵 |

---

## 5. 实施步骤（建议顺序）

1. `PreadExpertStreamer` 加 `useMmap` 数据面（mmap init / 专家 buffer 缓存 / mincore 规划 / madvise 预取 / munmap deinit），公共 API 不变。
2. `ExpertStreamingMode` + `Model.swift` 接线 + `TURBO_FIELDFARE_EXPERT_MMAP=1`。
3. telemetry 扩展（mincore 口径 hit、fault 计数）。
4. 交错 A/B（64 槽 vs mmap，slots=16/64 两组，r3+r4，msg_code 固定 prompt，6 轮中位）：tok/s、io bytes、hit%、fault%。
5. MTP A/B（slots 无关后，draft=3）复核 −10~15% 是否转正。
6. 输出结论并决定默认值。

**改动量估计**：数据面 ~200-300 行（集中在 streamer 一个文件），接线 + telemetry ~50 行，env 默认 off。风险集中在 GPU 直读页缓存这一新路径，用 A/B + 输出一致性校验兜底。

---

## 6. 诚实边界（mmap 不解决什么）

- **不同专家首次触碰仍需从盘读一次**（专家并集 IO 的增长不是「拷贝」而是「磁盘字节」）；mmap 消除的是重读/重拷与槽颠簸，不是首读。
- 页缓存被其他负载挤占时（内存压力），仍会抖动——42Gi 空闲下暂不构成问题。
- 方案不改变量化、不改变 kernel；是纯 IO 数据面替换。

---

## 7. 实测结论（2026-08-07 实施后）— mmap 在 16GB Mac 上不敌 pread

已完整实现 `TURBO_FIELDFARE_EXPERT_MMAP=1` 数据面并逐项实测，最终结论：

### 7.1 实现内容（全部已验证正确）

- **单 stream buffer + per-expert offset**：每层一个 buffer 包住整条 expert stream（`bytesNoCopy`），arg buffer 用 `setBuffer(buffer, offset:)` 编入 per-expert 偏移（MSL kernel 零改动），slot 表只做 hit/miss 记账
- **WILLNEED + 逐页 touch**：`POSIX_MADV_WILLNEED` 批量预读 + 每 16K 页读 1 字节强制驻留。touch 是**正确性硬要求**——GPU 无法 fault 真正冷的 file-backed 页（会读成垃圾）；早期 blit 探针"证明冷页可读"是假象：文件刚被自己写入，页全在缓存里
- **useResources 去重**：所有 blob 指向同一 stream buffer → 每次 CB 只注册 1 次驻留
- **arg buffer 时序修复**：phase2 的 split argBuf 从 post-fetch 的 fresh blobs 重新编码（否则 miss 槽的地址/偏移是加载前的旧值 → 垃圾）

### 7.2 正确性（达成）

temp-0 固定 prompt 下 mmap 与 pread 输出**逐字节一致**，多次运行稳定。

### 7.3 性能（未达成，且无法达成）

| 指标 | pread | mmap（健康时） | mmap（压力下） |
|---|---|---|---|
| decode | 11.8-14.3 tok/s | 9.1-10.3 | **卡死/爬行** |
| TTFT | 2.6-2.8s | 3.8-4.2s | — |
| IO（readWall） | 1.2s | **0.06s** | — |
| CB 编码开销 | 0.97s (78μs/cb) | 4.3-4.9s (350-400μs/cb) | — |

三个不可逾越的障碍（均为平台行为，非实现缺陷）：

1. **file-backed `useResource` 驻留税**：Metal 对 `bytesNoCopy` file-backed buffer 做同步驻留 walk，~85μs/buffer/CB。批量/去重只省调用开销，驻留工作按 buffer 计。IO 省的 1.2s 远小于税花的 3.4s → **mmap 结构性赢不了 pread**
2. **TTFT +1.2s**：prefill 首触专家时 WILLNEED+touch 在关键路径上逐个 fault
3. **间歇性 GPU 页回收卡死**：16GB RAM 压力下 clean file-backed 页随时被内核丢弃，GPU 的映射失效 → 逐页慢速 re-fault，decode 从 2 token/90s 到完全卡死（多次复现）

### 7.4 结论与后续

- **pread 保持默认**（未受影响，逐字节回归验证通过）；mmap 保留为 opt-in 实验路径（`TURBO_FIELDFARE_EXPERT_MMAP=1`；128-slot 全常驻用法：`TURBO_FIELDFARE_EXPERT_MMAP=1 TURBO_FIELDFARE_EXPERT_SLOTS=128`）
- 用户原目标「mmap 赢 pread → 128 slots 全常驻 → MTP 转正」在这台 16GB Mac 上被 Metal+file-backed 语义挡住
- **MTP 常驻的正确路径**：
  - 首选：72GB L20N 服务器上做**匿名内存全量常驻**（10-13GB 专家池一次加载，匿名 `bytesNoCopy` 无驻留税、无回收风险）→ MTP 正收益如报告预测（r4 +23~64% / r3 +52~103%）
  - 次选（16GB Mac）：**匿名热子集池**——每层 top-N 高频专家（~2-4GB）常驻匿名内存，其余流式；MTP verify 的并集主要落在热子集 → 大部分收益、内存可控

---

## 8. 匿名热子集常驻池（hot pool）— 已实现并实测

mmap 路被平台挡死后，按 7.4 的次选路线实现了 **profile 驱动的 pinned 匿名热池**（`pread` 模式的纯增强，不占额外内存——从 slot 预算中划出）：

### 8.1 实现

- **profile**：从一次 trace（`TURBO_FIELDFARE_EXPERT_TRACE`）统计每层 top-N 高频专家 → JSON（按层索引）
- **pinned 池**：streamer init 时把 top-N 专家 pread 进专属 slot 并标记 `slotPinned`，planner 的 evictable 列表永久排除 pinned slot（永不移除、永不重读）
- **预算**：poolSize = min(N, slotCount − 16)，保证 ≥16 个 LRU slot 能放下单 token topK 与 MTP 并集（≤16）
- 用法：`TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=32 TURBO_FIELDFARE_HOT_POOL_PROFILE=/tmp/hot_pool_profile.json`

### 8.2 覆盖率（trace 实测，30 层）

| 池大小/层 | 请求量覆盖 | MTP verify 并集覆盖 | 内存（匿名） |
|---|---|---|---|
| top-16 | 66.5% | 56.9% | 1.6GB |
| top-32 | 87.4% | 82.5% | 3.2GB |
| top-48 | 95.0% | 92.6% | 4.8GB |

### 8.3 实测（r3，64 slots，temp0）

| 配置 | code prompt | prose prompt |
|---|---|---|
| base（无池） | 11.79 tok/s | 12.16 |
| **pool** | **14.79（+25%）** | **12.89（+6%）** |
| base+MTP (82% accept) | 11.78（MTP 中性） | 7.58 |
| pool+MTP | 11.50（−22% vs pool 非 MTP） | 7.08（+7.6% vs base MTP） |

- **池是真实的 decode 收益**：code prompt +25%（3 轮全胜）、prose +6%——pinning 防止高频专家被 LRU 挤出重读
- **MTP 的 IO 半壁被池解决**：readWall −10%、bytes 8.55→8.28GiB；但 **MTP 仍中性偏负**——batched verify 的 GPU 计算（draft + 4 行前向）是主导成本，池管不到
- 结论修正：**「常驻 → MTP 转正」只解决了 IO 半边；GPU 计算半边需另找（draft 复用共享 FFN、verify 稀疏化）或接受 MTP 在该硬件上的中性定位**

### 8.5 「17.5% 是否拖累」— 决定性实验（pool48 = 92.6% 并集覆盖）

code prompt, 64 slots, 82% accept：

| 配置 | decode | tok/s | readWall | bytes |
|---|---|---|---|---|
| pool32 无 MTP | 8.32s | 15.38 | 4.16s | 9.86GiB |
| pool32 + MTP | 11.09s | 11.55 | 4.22s (+0.06s) | 8.28GiB |
| pool48 + MTP (92.6%) | 10.95s | 11.69 (+1%) | 3.86s (−8%) | 9.73GiB |

结论：
- 覆盖 82.5%→92.6%（尾清零）→ tok/s +1%（噪声内）→ **非池 17.5% 不是拖累**
- 池相同时 MTP 开 vs 关：IO +0.06s、bytes 反降，但 decode +2.77s（+33%）→ **MTP 成本 = batched verify 的 GPU 前向计算**，与专家 IO 完全解耦
- 即使 100% 常驻，verify 计算照样在 → **MTP 在此硬件上不会因常驻转正**；转正只能砍计算（draft 复用 shared FFN / verify 稀疏化 / 减 draft 数）

### 8.4 代码状态

- 改动：`PreadExpertStreamer`（pinned 池 + planner 排除）、`Model`（env + profile 解析）、无 MSL 改动
- 正确性：pool on/off 输出逐字节一致、多次稳定；默认（无池）行为零变化
### 9. B3 — decode attention tensor-ops 移植:可行性評估(2026-08-07)

#### 9.1 現狀測量(r3, 64 slots, temp0, code prompt, 128 tok)

```
[gpu] attn=2.03s routedMoE=1.70s sharedFFN=0.79s phase1Hit=0.32s head=0.83s busy=5.67s ofWall=44%
[sync] cbs=12760 cb1Wait=5.60s sched=7.32s overhead=0.87s(68us/cb) ofWall=45%
[stage] cb1=0.16s(1%) io=2.21s(16%) gpuWait=10.06s(75%) wall=13.49s
decode = 9.23s (128 tok, 72ms/step)
```

- **attn = 2.03s = GPU busy 的 36%,最大單項**;但 GPU 只佔 wall 的 44%,CPU 側 sched/idle(7.3s)才是更大的洞
- 30 層中 **5 層 full-attention**(512/16/2,indices 5/11/17/23/29)、25 層 SWA(256/16/8,window 1024)
- 短上下文下 full ≈ 29% 的 attention 位置工作量(≈0.6s);**長上下文(≥4K)full attention 無界增長,SWA 封頂 1024 → full 佔 60-76%**,價值隨上下文增長

#### 9.2 既有藍圖(已驗證、已是 M4 生產路徑)

- `prefill.metal` 已有 `attention_prefill_full_tensorops_2d_validity_v2`:mpp::tensor_ops `matmul2d`(8 輸出 × 512 head-dim,keys 以 64 為 tile)+ flash rescale(row_max/row_sum/row_old_scale)
- OPTIMIZATION_JOURNEY 記錄:8 head 一次處理,**64K 下 attention 11x**、32K prefill 491→204s(2.4x);reduce order 微改 logits 已被證實無害
- decode full attention 與 prefill 是同一數學(queryCount=1 的特例);目前 decode 走 per-position 純量 FMA + 每位置 ~4 barrier(GQA-full 1024 threads)+ flash-decoding 16-chunk 兩趟(partial+combine)

#### 9.3 移植方案

- 新 kernel `attention_decode_full_tensorops` = prefill tensor kernel 的 queryCount=1 特化:
  - grid.x = 1(query)、grid.y = numQHeads/8 = 2(KV-head group),128 threads
  - kvValidCount = seqLen,keys 以 64 為 tile 單趟 flash,直接寫 attnOut(省 combine 趟)
  - 只服務 full 512/16/2、ringCapacity==0、scale==1.0 的層;SWA 層不動(後續需 256/16/8 變體)
- env gate `TURBO_FIELDFARE_ATTN_TENSOROPS=1`(默認 off,與其他實驗 knob 一致),A/B 定案後再決定 default

#### 9.4 預期收益與風險

| 項目 | 預期 |
|---|---|
| full 層 attention kernel 時間 | MPP 8-head 並行 + 64-key tile(4 barriers/64 keys vs 4/position)→ 估 2-3x(短 ctx)~10x+(長 ctx,同 prefill 數據) |
| 短 ctx 總 decode | attn 0.6s 部分 → 省 ~0.3s ≈ 3%(有限) |
| 長 ctx(4K+)| full attention 60-76% → 省 20-30% decode,隨 ctx 增長 |
| CB 數 | 不變(仍在 cb1 內) |

- **數值風險**:f16 MPP 累加順序不同 → logits 可能微差(與 prefill 同源,已證無害);必須逐字節驗證
- **並行風險**:單趟 2 TG 失去 flash-decoding 的 chunk 並行;長 ctx 若變慢需改 chunked-tensor 變體(phase 2)
- **覆蓋風險**:SWA(25/30 層)本輪不動

#### 9.5 驗證計劃

1. 構建 → decode tensor 路徑 on/off 輸出**逐字節一致**
2. `[gpu] attn` 對比(短 ctx 128 tok)+ 長 ctx(600+ tok)對比
3. 3-way 交錯 A/B(同機、防頁緩存漂移),取中位

### 9.6 B3 Phase 1 實作結果(2026-08-07)

**實作**:decode `Attention.swift` 複用 prefill tensor-ops kernel(queryCount=1),env `TURBO_FIELDFARE_ATTN_TENSOROPS=1` gate(默認 off),只服務 full 512/16/2 層,單趟直寫 attnOut(省 partial+combine)。零 MSL 新增。

**正確性**:tensor on/off 輸出**逐字節一致**(4211ccfe),多次復現。f16 MPP 累加與 tiled 路徑在 argmax 層面零分歧。

**A/B(6 輪交錯,中位)**:

| 場景 | base attn | tensor attn | base tok/s | tensor tok/s |
|---|---|---|---|---|
| 短 ctx(163 pos,128 tok) | 2.81s | 2.82s | 10.41 | 10.28 |
| 長 ctx(~600 pos,600 tok) | 11.83s | 11.34s(−4%) | 15.22 | 15.97(+5%) |

**結論:短上下文中性、長上下文微正(噪聲內)**。原因:

1. **decode 只有 1 個 query → 2 TG 的 MPP 利用率極低**;prefill 的 11x 紅利來自大量 query TG 並行,decode 沒有
2. **tiled flash-decoding(16 chunk × 2 KV = 32 TG 並行)已足夠好**;單趟 tensor 2 TG 串行掃 key-tile 反而損失並行度
3. **full 層只佔 5/30**(short ctx ≈29% attn work)→ 任何 kernel 收益被稀釋 3-6 倍
4. **attn 只佔 wall 的 ~22%**(2.8s/12.8s);即便 attn 2x 也只省 ~8% wall;真正的洞是 CPU 側 sched(7.3s)

**Phase 2(若要繼續)**:chunked tensor——把 KV 範圍切成多段、每段一個 TG 跑 tensor kernel + 現有 combine 合併。這才能複製 prefill 的並行紅利;且只有上下文 >1024(SWA 封頂)後 full attention 佔比才 >50%。當前保留 env-gated opt-in,默認不動(tiled),零回歸。

### 10. B4 — CPU 側 sched/idle 深挖:多 queue 併發能省多少(2026-08-07)

#### 10.1 問題

decode 12.8s wall 中 GPU busy 只有 5.65s(44%),其餘是 CPU 側 sched/idle(12,760 個 CB)。假設:多 MTLCommandQueue + events 併發能填補 GPU 飢餓。

#### 10.2 新儀器:[cb-latency](GPU_TIMING gate)

per-cb1 分解(commit→gpuStart→gpuEnd→waitReturn),加入 `RealForwardRunner` + `[cb-latency]` diag 行:

```
[cb-latency] wait=7.24s gpu=2.68s wake=0.61s sched=4.03s other=0.00s ofWall=48%
```

- `wait` = waitUntilCompleted 總時長(每層一次,30 次/step)
- `gpu` = cb1 的 GPU span
- `wake` = gpuEnd → waitReturn(執行緒喚醒)
- `sched` = commit → gpuStart;實測 ≈ 前一層 routed+shared+phase1Hit 的排隊 GPU(3.8s)+ 真 dispatch 延遲(~0.2s,68-72us/cb)
- `other` = 0 → wait 完全被分解,無殘留

#### 10.3 多 queue 結論:≈ 省不到東西 — 依賴鏈是嚴格鏈

1. **沒有可並行的 GPU 工作**:每層 cb1 需要前一層 routed 的 GPU 輸出(hidden);routed 需要 cb1 的 router readback。第二條 queue 沒有獨立工作可跑
2. **GPU idle 4.04s 的歸因(`[gpu-idle-after]`)**:cb1→shared ≈0(shared 已 pre-commit)、routed→cb1 ≈0(cb1 已 pre-commit)、**shared→routed = 2.12-2.30s + phase1Hit→routed = 1.60-1.76s → 3.7-3.9s 是 GPU 等 expert fetch / routed commit**
3. **hot pool 驗證**:idle-after-sharedFFN 只從 2.30→2.12s(池已把 miss 壓到 ~13%)→ 殘餘 idle 是主執行緒 readback→plan→fetch-await→routed-encode→commit 的串行 CPU 下限(~0.6ms/layer ≈ 18ms/step),不是磁碟
4. **wake 實驗(continuation vs waitUntilCompleted)**:0.59s vs 0.59s **完全一樣** → wake 是 Metal 內部 completion 成本,執行緒阻塞方式無法縮減;已回退
5. **結論:MTLCommandQueue 併發對單 stream decode 延遲沒有可兌現收益**;GPU idle 是 IO-bound(fetch)與主執行緒串行工作,不是 queue 調度問題

#### 10.4 真正有效的槓桿(按 ROI)

| 槓桿 | 狀態 | 預期 |
|---|---|---|
| hot pool(固定高頻專家) | ✅ 已做(code +25% / prose +6%) | 直接縮小 fetch → idle |
| 更大的池(top-48 = 92.6% 覆蓋) | 選項 | 更多同向 |
| 主執行緒串行下限(0.6ms/layer) | 需深層重構(routed encode 移背景執行緒) | ~15-18ms/step,風險高 |
| wake 0.61s(5%) | 實測不可縮 | 0 |
| B1 CB 融合 | 已測回歸 | 0 |

#### 10.5 程式碼狀態

- 新增:cb1ScheduleNanos/cb1WakeNanos 計數器 + `[cb-latency]` diag 行(GPU_TIMING 才啟動,默認路徑零影響)
- 實驗:continuation wait(TURBO_FIELDFARE_CONT_WAIT)實測中性,已回退
- 正確性:默認路徑輸出逐字節一致

### 11. 三個候選優化的實測定案(2026-08-07)

#### 11.1 A1/A2 — prefill 並行讀:中性(SSD-bound,非讀取模式問題)

247-token prompt,交錯 3 輪中位 TTFT:

| 模式 | workers=2 | workers=8 |
|---|---|---|
| baseline | 9.07s | 9.24s |
| coalesced | 9.06s | 9.12s |
| layerLocalReadahead | 9.05s | 9.16s |

- 三種模式全在 ±3% 噪聲內;workers 8 反而略差
- 根因:prefill 讀取是 **SSD 吞吐 bound**(1.79-1.92GiB/s 穩定),coalesced/readahead 改不了磁碟吞吐
- **真正的 TTFT 槓桿是 hot pool**:同 prompt TTFT 9.19→7.87s(**−13%**)— prefill 讀取命中 pinned 專家
- 結論:A1/A2 模式保留為選項(server 場景可能不同),CLI 默認 baseline 不動

#### 11.2 桌面 CPU 搶佔:無安全可關目標(iCloud daemon 突發,已自行消退)

- 真兇:bird 30% + ContainerMetadataExtractor 26% + knowledge-agent 25% + coreduetd 21%(系統 iCloud 同步 daemon 突發)
- WindowServer 31%(必需);WorkBuddy/Freebuff(會話本身);模型檔非 fileprovider-backed
- 突發已自行消退(bird 30%→4.3%)→ tok/s 仍 ±30% 擺動是系統層噪聲(load 2.0+,4 users),非 App 可關
- 結論:沒有可安全關閉的桌面 App;等待 iCloud 同步完成即可

#### 11.3 routed encode → 背景執行緒:不值得做 — 修正先前估計

- **cb2(encode+commit)只有 0.07s/run = 0.7ms/step(0% of wall)** — 不是串行下限
- 串行下限是 expert pread:長 prompt readWall 5.78s = **37.5% of wall**、ofDecode 78%,SSD-bound 1.79GiB/s
- GPU idle 2.74s = sharedFFN 1.56s + phase1Hit 1.18s(等 fetch),不是等 encode
- 修正:先前「~18ms/step 串行下限的 60-70%」估計錯誤;encode 移背景執行緒最多省 ~0.7ms/step(1%)
- **真正剩餘槓桿**:更大的池(top-48,92.6%)、2-bit/3-bit 權重(讀取 bytes 減半,已 A/B)

### 12. 生產配置固化 + top-48 池實測(2026-08-07)

#### 12.1 top-48 池實測(code + prose,交錯中位,load 3.36 噪聲窗口)

| 配置 | code ttft | code tok/s | prose ttft | prose tok/s | RSS |
|---|---|---|---|---|---|
| nopool | 3.51-3.57s | 12-14 | 4.16s | 12.7 | ~2.7GB |
| pool32 | **3.02-3.05s** | 12.4-13.5 | **3.60s** | 9.6 | ~3.0GB |
| pool48 | 3.40-3.44s | 15.1(高窗 16-18) | 3.94s | 10.1 | ~3.0GB |

- **RSS 意外收穫:池幾乎零額外內存** — pool 用既有 slot buffer(64 slots × 30 層已分配),只是 init 預載,pool48 只 +288MB
- **pool48 decode ≥ pool32**(乾淨窗口 16-18 tok/s vs 14.8),但 **TTFT +0.35s init 預載稅**(4.8GB pread vs 3.2GB)
- 噪聲窗口下 prose 無法定案(code profile 對 prose 覆蓋本就較低)
- 正確性:pool48 輸出逐字節一致;roundtrip profile == shipped profile

#### 12.2 生產配置(已交付)

- **`bin/run_prod.sh`**:包裝全部驗證過的設定
  - `TURBO_FIELDFARE_EXPERT_SLOTS=64`(命中 95%)、`HOT_POOL=1` + profile(默認 48,`HOT_POOL_PROFILE_SIZE=32` 可切)、`--trust-receipt`(TTFT −3.9s)
  - adaptive MTP / early shared 已是 code 默認開
- **`bin/make_hotpool_profile.sh`**:一鍵生成 profile(trace → gen_hotpool_profile.py → 與模型同目錄 `profiles/top<N>_code.json`)
- **`Scripts/gen_hotpool_profile.py`**:profile 生成器(已入 repo)
- **`models/profiles/top32_code.json` / `top48_code.json`**:shipped profiles
- 驗證:prod 路徑輸出逐字節一致、roundtrip 可重現、錯誤路徑有清晰提示

### 13. 非阻塞池預載(TURBO_FIELDFARE_HOT_POOL_PRELOAD=async,2026-08-07)

#### 13.1 動機

pool48 的 init 同步預載 4.8GB 阻塞在 prefill 首觸路徑(streamer lazy 建立),TTFT +0.35s 稅。

#### 13.2 實作

- init 時**先 pin 池 slot**(slotPinned=true、slotOwnerPhase=.sharedResident,零讀取)→ planner 永不移除/分配
- slotExpert 保持 -1 直到資料落地 → 未載入的池專家只是普通 miss(載入 LRU slot),池「尚未熱」而非錯誤
- 資料由後台任務載入(2-deep 併發 semaphore + utility QoS,跨 30 層不 thrash SSD),完成後 cacheLock 下寫 slotExpert/hitCount
- 任務持強引用 → streamer/slotPointers 不會提前釋放
- env `TURBO_FIELDFARE_HOT_POOL_PRELOAD=async`,默認 sync 保持原行為

#### 13.3 A/B(code prompt,4 輪交錯中位)

| 指標 | sync | async | Δ |
|---|---|---|---|
| TTFT | 3.37s | **3.02s** | **−0.35s(稅全回收,4/4 輪一致)** |
| decode | 12.31 | **14.79** | **+20%** |

- 256-token 驗證:decode 命中率 **94.7%**(池在 decode 期間填滿並生效)、TTFT 3.06s 保持
- 正確性:sync/async 輸出**逐字節一致**
- 額外:async 下 readWall 吞吐 2.18GiB/s(>sync 的 1.9),背景預載順帶利用 decode 的空閒 SSD 窗口
- run_prod.sh 已默認 async;`TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync` 可回退

### 13.1 MTP_MODEL 旋鈕接入 run_prod.sh(2026-08-07)

- `bin/run_prod.sh` 新增 `MTP_MODEL` env 旋鈕:
  - 默認:`/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head`(MTP 開)
  - `MTP_MODEL=""`(注意:空字串,非 unset)→ MTP 關,純 base decode
  - 用的是 `${VAR-default}` 而非 `:-`,讓「空字串=關閉」與「unset=默認」可區分
- 腳本同時拒收 caller 傳 `--mtp-model`(與 `--model` 同規則,避免雙源分歧)
- bash 3.2 + `set -u` 空陣列展開 bug 已用 `${MTP_ARGS[@]+"${MTP_ARGS[@]}"}` 慣用法修復
- 接上後 **adaptive gate 自動生效**(code 默認 ON,`TURBO_FIELDFARE_MTP_ADAPTIVE=0` 可關):
  - 實證 footer:`mtp=10/12(83%) adaptive(d=0 red=0 off=0 row=...)` — MTP 與 gate 同時活躍
- 128-token 6 輪交錯 A/B(code prompt,load 2.7 噪聲窗口):
  - base 中位 **12.90 tok/s** vs MTP-adaptive 中位 **15.31 tok/s** → **+18.6%**
  - session 歷史最佳(code):adaptive 14.4 vs base 11.5 → +25%
  - prose:gate 低接受時自動 d=0 走 decode path,≈base 不拖累(結構保證)
- 結論:code 類 prompt +15~25%,prose ≥0%(gate 保證不劣於 base),TTFT 不受影響

### 13.2 MTP-adaptive 接受率 + verify 可省性審計(2026-08-07)

**現狀(code prompt, 實測)**:
- 接受率 **84%**(27/32 drafted),probe 窗滾動接受率 100% — draft 品質本身健康
- tok/s 15.4(乾淨窗),+18.6% vs base(6 輪交錯中位);decode 緩存命中 95.9%
- **gate 過度禁用**:`TURBO_FIELDFARE_MTP_ADAPTIVE_DEBUG=1` 逐 step 顯示 — calib 後 probe(d=2,acc=1.00)跑 8 步即禁用,剩餘 ~60% 步數走 d=0 decode path,全程 `stepsAtMaxDraft=0`
- 根因:8-step 窗口 + ±5% 帶 + hardLow=-10% 在機器 ±30% 噪聲下統計無意義;MTP 步 >3.33× base 步即誤判禁用。clean 窗 A/B(歷史)adaptive 保持 d=3 吃到 +25%,噪聲窗則大幅禁用以致 +18.6%

**verify 結構審計**:
- 已最優:單 forward batch 驗證全部 draft(1+D 行,`verifyBatch→prefillChunked` 單 span);pool 讓 verify expert 並集共享;gate 自動調 draft 數
- 兩段式 verify(先驗 draft[0] 再 batch 其餘):α=0.84 數學省 ~12% verify 行,但加一次 forward 固定開銷(~85μs CB + readback),淨收益≈0,不建議
- verify 層稀疏(batched forward 下只驗必要層):架構不可行(所有行共享層管線)
- 真槓桿:**draft 品質**(84%→90%+ 直接多 ~0.5 token/cycle,零 runtime 成本,需 fine-tune MTP head)與 **gate 窗口穩定化**

**總體剩餘提升(依 ROI 排序)**:
1. gate 窗口穩定化(windowSize 8→32、rate EMA、禁用前多輪確認)→ 高接受率 run 全程吃 draft,+18.6%→+25%
2. MTP head fine-tune(接受率↑,零 runtime 成本)
3. B3 Phase 2 chunked tensor attention(長 ctx)
4. B1 CB 融合(8808 CBs,330us/cb)
5. 2-bit/3-bit 權重(讀取減半)

### 13.3 五項優化全推(2026-08-07) — 進度與實測

**① gate 窗口穩定化 — 已實作並驗證**
- MTPAdaptive.swift:decideEvery 8→16、新增 rateEMA(emaAlpha 平滑原始窗口速率)、禁用/降級需**連續 2 次輸判定**(單次噪聲窗口不能誤殺高接受率 prompt)
- 驗證(DEBUG trace):code prompt off=10→**off=1**,接受率 27/32→**70/80(88%)**,全程保持 draft;交錯 A/B(噪聲窗)+3.8%(EM A 更保守,但不再誤判禁用)
- 剩餘:EMA 使升檔更保守(停在 d=2),乾淨窗會升到 d=3-4

**② MTP head fine-tune — 審計:屬雲端工作流,Mac 端受阻**
- `app/training/train_mtp.py` 管線完整(collect_hidden_states → distill_train → validate_mtp_accept),但面向 **sglang + CUDA + /data/models**(host1)
- Mac 端缺:①本地無 HF gemma4 base(collect 需要)②collector 用 `CGC_Phase2.mtp_verify_loop`(sglang 遠端)③訓練產出 `.pt` checkpoint 無 →Metal drafter(safetensors)轉換器
- 正確執行位置:host1(`/data/models/gemma-4-E4B-it` + CUDA),需補 `.pt→safetensors` 導出器才能餵回本地 Metal drafter

**③ B3 Phase 2 chunked tensor attention — 未做(長上下文專屬)**
- Phase 1 已證:短 ctx(163pos)中性、長 ctx(~600)tok/s +5%;SWA 封頂後(>1024)full attention 才 >50% 佔比
- 設計:把 prefill tensor kernel(MPP matmul2d)的 KV 切段、每段一 TG + 現有 combine,複製 prefill 的並行紅利
- 當前 prompt(35-600 tok)無收益,列為長上下文工作負載的後續項

**④ B1 CB 融合 — 審計:已完成,無殘餘**
- decode loop 現況:每層 2 CB(cb1 = GQA-full 單 kernel attention + router + lookahead;sharedCB early-commit;routed 非同步),8808 CBs/128tok ≈ 2.3/layer — 原 12,416 CB 是舊內核
- `TURBO_FIELDFARE_FUSE_SHARED`(shared 併入 cb1)已在乾淨機器實測**回歸**、默認 off:併入會延遲 router readback → routed 晚啟動,破壞 CPU/GPU 重疊

**⑤ 2-bit/3-bit 權重 — 已實作並實測(r2 repack + A/B)**
- `TurboFieldfareRebits --input gemma4.gturbo --routed-bits 2` → `models/gemma4-r2.gturbo`:packed 12.9→7.2GB(−44.4%),權重空間 RMSE gate 0.0125/up 0.0127/down 0.0190
- 6 輪交錯 A/B(code,128tok):**TTFT 3.56→2.25s(−36.7%)、tok/s 11.07→18.57(+67.7%)**
- 品質:固定 prompt 對照 — r2 輸出正確且更工整的 fibonacci memoization(無退化)
- **r2 上 MTP 接受率降**(46% vs r3 77%):量化偏移 target 分佈,draft head 對齊度下降
- `run_prod.sh` 新增 **MODEL_BITS=2|3**(默認 3,`TURBO_FIELDFARE_MODEL` 仍優先);`MODEL_BITS=2` 生產路徑驗證通過(TTFT 2.2s)

### 13.4 審查修正 + r2 profile 澄清(2026-08-07)

- gate 審查 2 個問題已修:①移除 write-only 死狀態 `winningDecisions` ②revive 分支會比對**凍結的舊 EMA** — 改為 probe 啟動時 `rateEMA=0` 重置,revive 用新鮮 probe 窗(避免 stale-high 誤復活/stale-low 永不復活)
- 第一決策仍無 EMA 平滑(seed=單樣本),真正改善來自 window 8→16;禁用需連續 2 輸 → 主題轉移最壞 ~32-48 cycle 才關(可接受,bounded)
- **r2 profile 澄清**:`run_prod.sh` 的 `$(dirname "$MODEL")/profiles` 解析到 `models/profiles/`(**共享目錄**,非每變體目錄)——r2/r3 同架構、re-quantization 保留 expert ID,r3 的 top48 profile 對 r2 有效;`MODEL_BITS=2` 生產路徑驗證通過
- gate 最終驗證:code prompt off=10→**off=1**,接受率 **88%**(70/80),無誤判禁用;draft 停在 2(EMA 保守,乾淨窗升 3-4)

### 13.5 Perplexity 評估工具 + r2/r3/r4 實測(2026-08-07)

**新增 `--perplexity <corpus.txt>` 模式**(TurboFieldfareCLI):
- `Args.swift` + `Run.swift` + `Command/main.swift`:logits head(forceLogitsHead)+ 逐 token produce + 讀 [vocab] FP16 logits buffer,logsumexp softmax 算 NLL
- 輸出:`ppl`(幾何均)/ `medPPL`(魯棒中位)/ `meanNLL` / `medNLL` / `p95NLL` / `argmaxAcc`(argmax 命中率)
- 注意:instruct model 對 raw prose 是 OOD(無 chat template 時 medPPL 高達 ~400K、近均勻)——**要套 chat template 才有意義**
- 驗證:重複文本 ppl=1.62(測量正確);fix 過程踩到 Args.swift 的 `modeMissing` 校驗(--perplexity 需豁免)+ repo 增量緩存坑(需 `swift package clean`)

**三變體 ppl(chat-templated corpus,178 tokens)**:

| variant | medNLL | medPPL | argmaxAcc |
|---|---|---|---|
| r4 | 9.43 | 12,464 | 23.0% |
| r3 | 10.78(+1.35) | 48,273 | 23.6% |
| r2 | 12.08(+2.65) | 175,989 | 16.3% |

**三變體 TTFT/decode(生產配置,4 輪交錯中位,load 3.6 噪聲窗)**:

| variant | TTFT | decode tok/s |
|---|---|---|
| r4 | 4.15s | 10.69 |
| r3 | 3.62s | 14.63 |
| r2 | 2.72s | 14.36 |

- r4 歷史最佳:TTFT ~4.0s(--trust-receipt;含全量 hash 時 8.05s)、decode **15.9 tok/s**(乾淨機,§8 B1 交錯中位);優化前基線 6.82s / 8.35
- 修正:13.5 首測的「r4 10.69」是三方交錯時被 r2/r3 頁快取串擾的低值;r4-only 6 輪實測中位 12.28 / 峰值 15.34(load 3.3 噪聲窗)
- r2 乾淨窗峰值 decode 18.5 tok/s、TTFT 2.72s(讀取減半)
- 結論:r2 付 medPPL ×14 換 decode +34% / TTFT -34%(vs r4);品質敏感場景用 r3,速度優先用 r2

### 13.6 top-64 池 + sync preload(2026-08-07)— MTP 在 r4 的最終裁定 + pool64 意外勝利

**動機**:測試「r4 + pool64 + MTP-adaptive」三方對照,確認池補足 IO 後 MTP 在 r4 是否轉正。

**top-64 profile 生成**:覆蓋率 **97.1%**(vs top-48 的 92.5%),profile 為 r2/r3/r4 共享(同架構、expert ID 保留)。

**第一輪 A/B 陷阱(重要教訓)**:pool64 初測崩到 5.3 tok/s、swap 14.1/15.36GB — 一度誤判為「96 slots × 3.2MB × 30 層記憶體爆掉」。鑑別實驗證明**根因是 async preload 的 6.1GB 背景讀取在 decode 期間與 miss 搶 SSD 頻寬**(2 並發 semaphore 把讀取攤到整個 decode),不是記憶體(posix_memalign 是 lazy commit,RSS ~2.5GB 安全)。swap 高企是 18-run 超時測試的惰性殘留,非即時瓶頸。

**r4 對照(4 輪交錯中位,64 tok,code prompt)**:

| 配置 | TTFT | decode | 總帳(64 tok) |
|---|---|---|---|
| pool48 + async | 3.84s | 9.04 tok/s | 10.92s |
| **pool64 + sync** | 4.50s | **14.84 tok/s** | **8.81s(−19%)** |

- pool64-sync 單項最低 10.3 仍贏過 pool48 最高 9.6 — 定案級別
- 總帳延伸:32 tok −10%、128 tok −27%、256 tok −32%、512 tok −36%(sync 的 0.66s 稅完全被 decode 加速覆蓋)

**r3 對照(生產默認,4 輪交錯中位)**:pool64-sync **15.98 tok/s(+17.8%)**、TTFT +23.8%(0.75s 稅);64 tok 短生成總帳打平(7.89 vs 7.93s),長生成明顯贏。

**MTP 在 r4 的最終裁定:確認負收益,不轉正**。即使 SSD 乾淨、覆蓋率 97.1%:

| 配置 | decode | 接受率 |
|---|---|---|
| pool64 + sync base | 14.70 tok/s | — |
| pool64 + sync + MTP adaptive | 5.78 | 63% |
| pool64 + sync + MTP fixed(d=4) | 7.14 | 49% |

r4 接受率(63-72%)低於 r3(84%),verify 的 batched forward 計算成本壓不過 draft 紅利;且 MTP loop 的 d=0 步本身有 ~2× base 的序列化 sync 開銷。**MTP 不是 r4 的槓桿** — 保持 r3 上的「adaptive + pool」組合即可。

**交付**:
- `run_prod.sh` 支援 `HOT_POOL_PROFILE_SIZE=64`(32|48|64)
- 聯動:pool64 → `TURBO_FIELDFARE_EXPERT_SLOTS=96` + `HOT_POOL_PRELOAD=sync`(pool64 必須 sync,async 會崩 decode −50%);32/48 → 64 slots + async 不變
- 驗證:prod script 三路徑全通(64/sync、默認 48/async、非法 50 拒絕)

**一句話**:pool64 + sync 是 decode 之王(r4 +64%、r3 +18%),MTP 對 r4 定案為不轉正;async 只適合小池,pool64 上必須 sync。

**定案(§13.6 追加)**:r4 128-tok 4 輪交錯 — p64-sync **中位 14.34 / 峰值 16.65 tok/s(破 15.9 歷史紀錄)** vs p48-async 中位 10.87(+31.9%)、總帳 13.5 vs 15.5s(−12.5%)。`run_prod.sh` 默認 `HOT_POOL_PROFILE_SIZE` 改為 **64**(自動 96 slots + sync preload);MTP_MODEL= 關 MTP、HOT_POOL_PROFILE_SIZE=48 回退 async 均端到端驗證通過(默認 128 tok 實測 14.05 tok/s、MTP 接受率 88%)。

### 13.7 B3 tensor-ops attention 定案(2026-08-07)— 已存在、+7.9%、設為默認

**審計發現**:`TURBO_FIELDFARE_ATTN_TENSOROPS` 的 single-pass MPP tensor-ops decode attention **kernel 早已存在且完整接線**(Attention.swift `psoDecodeTensorOps`,512/16/2 full-attn shape),只是 env-gated 默認 off 且從未測過性能。

**A/B(r4 + pool64-sync,128 tok,4 輪交錯中位)**:tensor-ops **14.23 vs base 13.19 tok/s(+7.9%)**,TTFT 中性。byte-identity 驗證:code prompt + 2 個 prose prompt 全部 `md5` 一致。

**GPU 拆解(關鍵洞察 — 收益來源不是 GPU 加速)**:

| 指標 | base | tensor-ops |
|---|---|---|
| GPU attn | 2.91s | 2.93s(沒變) |
| cb1Wait | 10.06s | 9.55s(−0.5s) |
| overhead/cb | 277us | 230us(−17%) |

tensor-ops 把每層 partial+combine 兩次 dispatch 鏈合成 single-pass,**省的是 CPU 側 dispatch/schedule 稅,不是 GPU kernel 時間**。

**剩餘空間拆解(回答「16.65 之後還有什麼」)**:完整 stage/gpu/sync 拆解顯示:

```
[stage] cb1=0.2s io=1.8s cb2=0.1s head=1.3s gpuWait=19.3s(85%)
[gpu]   attn=2.7s routedMoE=2.7s shared=1.1s head=1.3s busy=7.9s ofWall=35%
[sync]  cbs=11957 cb1Wait=13.7s sched=9.9s overhead=7.3s(613us/cb)
[cb-latency] wait=13.7s gpu=2.7s wake=0.5s sched=10.6s(!!) other=0
```

- **最大單一成本是 sched=10.6s(CPU commit → gpuStart 排隊延遲),不是 attention** — 是「GPU 跑完一批等 CPU 提交下一批」的 pipeline 深度問題(B4 的獵物),遠超 attention 的 2.7s
- GPU busy 只有 35-47% of wall — **GPU 有一半以上的時間在等**,調度空轉才是主矛盾
- attention 2.7s 已不是最大 GPU 單項(與 routedMoE 2.7s 並列),B3 的收益已兌現(+7.9%)

**交付**:`run_prod.sh` 默認 `TURBO_FIELDFARE_ATTN_TENSOROPS=1`(opt-out =0)。prod 默認 128 tok 實測 11.84 tok/s(MTP 88% 接受率)。

**一句話**:B3 不值得「重寫」— 它早已存在,開默認 +7.9%;真正剩下的洞是 sched 10.6s 調度空轉(B4 pipeline 深度),不是 attention。

### 13.8 B4 pipeline 深挖 — hit-only sync fetch(2026-08-07)

**動機**:sched 10.6s 是最大單一成本。調查「提前提交下一層 cb1 / 多 CB 在飛」可行性。

**結構性結論:「提前提交下一層 cb1」架構上不可行** — 下一層 attention 依賴本層 routed 的 hidden 輸出,是硬數據依賴。decode 是每層 `commit cb1 → waitForCompletion(cb1) → router readback → plan → fetch → commit routedCB` 的嚴格串行鏈。

**真正的獵物**:GPU idle 拆解(gpu-idle-after)顯示 **sharedFFN 後 4.26s(3800 個 ~1.1ms 小 gap)** — 即使全 hit,每層 fetch 都走 `withCheckedThrowingContinuation + DispatchQueue.global.async` 的跨執行緒 hop。

**實作**:`fetchRoutedExpertsHitOnlySync` — plan.misses 為空(全 hit)時同步執行,跳過 continuation+dispatch hop。`TURBO_FIELDFARE_B4_HIT_ONLY_SYNC` 默認 ON(`=0` 關閉),審查後 guard 取代 precondition、加 ModelError case。

**A/B 結果 — 結構改善但吞吐中性**:

| 指標 | base | B4 | Δ |
|---|---|---|---|
| gpuIdle | 9.19s | 3.41s | −63% |
| overhead/cb | 597us | 162us | −73% |
| cb1Wait | 13.05s | 7.12s | −46% |
| >5ms 長 gap | 180x | 43x | −94% |
| **decode(4 輪交錯中位)** | 12.46 | 12.56 | **+0.8%(中性)** |

**關鍵洞察(矛盾即真相)**:消除 63% GPU idle 沒有轉成吞吐 — **decode 是 CPU 串行鏈瓶頸(latency-bound),不是 GPU 吞吐瓶頸**。每層 `waitForCompletion(cb1)`(GPU 完成 attention+router 才能 readback indices)是物理上不可省的一環;GPU 有閒置容量但 CPU 必須等。sched 排隊延遲的根源是串行鏈深度,不是提交節奏。

**保留決策**:審查 flag「off-by-default 是死代碼」。B4 設默認 ON(嚴格更簡單:相同工作、更少 hop、byte-identical、無風險);與 PRELOAD=async 的 cacheLock 交互已註記為 latent(prod 默認 sync preload)。

**一句話**:B4 證明「GPU idle 消除 ≠ 吞吐提升」— decode 的牆是 CPU 串行鏈(waitForCompletion 每層一次),不是 GPU。突破它需要真正的架構變化(如 MTP verify 的多 token 並行),而非更快的單步提交。

### 13.9 GPU span 時間線全分析(2026-08-07)— idle gap 的真面目

**工具**:Run.swift 新增 `TURBO_FIELDFARE_GPU_TIMELINE_CSV=<path>` — dump 完整 span 序列(start/end/dur/label/gap)。配套 `bin/analyze_gpu_timeline.py` 深度分析。

**數據(256 tok,pool64-sync+tensorops+B4,24086 spans)**:gpuBusy=14.97s gpuIdle=11.13s window=26.07s。

**gap 分布 — 不是「千個亞毫秒縫隙」也不是「幾個長停頓」,是「7650 個 ~1.2ms 規律縫隙 + 少數 miss 長停頓」**:

| bucket | count | 時間 |
|---|---|---|
| <0.2ms | 17755 | 0.20s |
| 0.2-1ms | 3695 | 2.02s |
| **1-5ms** | **2364** | **4.51s** |
| 5-20ms | 212 | 2.21s |
| 20-100ms | 59 | 2.09s |
| >100ms | 1 | 0.10s |

**正確歸因(修正:gap i 是 span i 之前的空閒 = span i−1 之後的空閒)**:

| 空閒位置 | n | 總時間 | 含義 |
|---|---|---|---|
| **after sharedFFN** | 7650 | **9.06s(81%)** | **每層唯一真縫隙**:CPU readback→plan→fetch→commit routedCB 鏈 |
| after phase1Hit | 626 | 1.74s | hit phase1 後等 miss pread |
| after routed | 7650 | 0.10s | routed→cb1 無縫(pipeline 重疊生效) |
| after cb1 | 7650 | 0.01s | cb1→shared 無縫(early shared commit 生效) |
| after head | 509 | 0.24s | token 邊界 |

**長停頓(>5ms)全部是 miss pread**:sharedFFN→routed 183x(2.97s)+ phase1Hit→routed 56x(1.00s),即 20-100ms 的 59 個長停頓 = miss 專家讀取。1-5ms 主力縫隙(1799x=3.40s)也是同一條 sharedFFN→routed 鏈,只是 CPU 快時較短。

**drift 測試**:sharedFFN→routed gap 隨上下文**縮短**(前 3 token med 1.46ms → 後 3 token 0.24ms)— 因為頁快取預熱後 CPU 側 fetch 更快,且 GPU attn 增長被 tensor-ops 消化。**不是 attn 增長問題**。

**一句話**:GPU idle 81% 是「每層 sharedFFN 完成後等 CPU 準備 routedCB」的規律縫隙(平均 1.2ms×7650)+ miss pread 長停頓。routed→cb1 與 cb1→shared 已完美重疊。這與 §13.8 B4 結論一致:縫隙是 CPU 串行鏈,不是提交節奏 — 唯一真正有效的下一個槓桿是 miss pread 的長停頓(2.09s,pool 再擴大)或打破串行鏈的多 token 並行。

### 13.10 最後兩個槓桿的證偽(2026-08-07)— pool80 與 prefetch 都不行

**前置修正**:時間線重算後,>5ms miss 停頓其實是 **4.40s(272 個,17% of wall)**,先前報 2.09s 是低估。>20ms 的 60 個長停頓 = 2.20s 是 miss pread。

**槓桿 1:pool 擴大到 top-80 — 記憶體牆**。top-80 profile 覆蓋率 99.1%(vs top-64 97.1%),miss 理論減半。但 r4 expert 3.2MB × 80 × 30 層 sync preload = **7.7GB commit** + 模型 12.9GB 頁快取 → 16GB 機爆。實測 128 tok 只有 6.63 tok/s、swap 持續 10.4GB。**pool64 是這台機器的記憶體極限**。注意:同窗 pool64 也掉到 8.75(Doubao 51% CPU 搶載),「pool80 失敗」需乾淨機才算最終定案,但 7.7GB commit 的記憶體帳本身就是硬牆。

**槓桿 2:expert prefetch(下一層 router 預測提前讀)— 與 hot pool 衝突**。`TURBO_FIELDFARE_EXPERT_PREFETCH` 默認沒開;4 輪交錯 A/B:prefetch **9.54 vs base 16.25 tok/s(−41.3%)**。根因:pool64 下 99% 已是 hit,prefetch 對已 pin 專家做重複 pread,浪費 SSD 頻寬。該機制是「無 pool 時代」設計,與 hot pool 正交且衝突。lookahead=None 也確認預測路徑未有效運行。

**結論**:兩個候選槓桿都被證偽。剩餘空間的真實分布(r4,pool64-sync+tensorops+B4):
- miss pread 4.40s:pool 再擴大不可行(記憶體),prefetch 衝突 — **只能靠 r3/r2 更小 expert 或接受**
- sharedFFN→routed 串行縫隙 9.06s:§13.8 B4 已證 CPU 串行鏈,單步優化已到極限
- **真正剩下的路:多 token 並行(MTP verify 泛化)或位寬切換(r2 = TTFT −37% / decode +68%)**

**生產默認不變**:pool64-sync + tensor-ops + B4(hit-only sync)+ adaptive MTP + trust-receipt。

### 13.11 EXPERT_READ_WORKERS=8 定案(2026-08-08,重開機乾淨窗)— +11.1% decode,零 TTFT 稅

**動機**:用戶提醒「多 worker 之前設定成 2」。查證:`TURBO_FIELDFARE_EXPERT_READ_WORKERS` 控制 decode+prefill 並行 miss pread 深度,默認 **2**(`boundedParallelMissReadWorkersDefault`),生產沒設。decode 每層最多 topK=4 個 miss,worker=2 時 SSD 深度不足。

**split-worker 實作**:`executeParallelMissReads` 加 `workerLimit` 參數;prefill 分支(含 MTP verify 的 prefillChunked)cap 在 `min(2, ...)`,decode 用完整 env 深度。理由:prefill 高並行會與 hot-pool sync preload 搶 SSD 頻寬(TTFT 稅)。

**重開機乾淨窗 A/B(swap=0,6 輪交錯,r4 pool64-sync+tensorops)**:

| 指標 | w2 | **w8-split** | Δ |
|---|---|---|---|
| decode med | 13.89 | **15.43** | **+11.1%** |
| best3 平均 | 17.0 | 18.1 | +6.5% |
| TTFT | 4.62s | 4.63s | **+0.1%(稅消失)** |

- **先前重載下的「中性」被證實是 load 污染**(±30% 噪聲);乾淨窗下 w8-split 每輪 ≥ w2
- MTP on 對照:10.68 vs 9.61(+11.1%),接受率同 72% — **verify 的 prefill-cap 無害**(batched 讀取量小)
- byte-identity 驗證 ✓

**審查閉環**:coalesced/readahead prefill 模式繞過 cap(opt-in 非默認,已理解);MTP verify 用 prefill plane 被 cap 已實測無回歸;magic number 改為複用 `boundedParallelMissReadWorkersDefault`。

**交付**:`run_prod.sh` 默認 `TURBO_FIELDFARE_EXPERT_READ_WORKERS=8`(opt-out env 覆蓋)。生產默認更新為:pool64-sync + tensor-ops + B4 + **w8 split-workers** + adaptive MTP + trust-receipt。

### 13.12 GPU busy 65%→85% 的串行鏈精確歸因(2026-08-08)— sched 不是洞,wake+chainWall 才是

**動機**:用戶要求把「GPU busy 65%→85% 的串行鏈縫隙」精確歸因 — sched vs wake vs cb1wait 各佔多少、哪些 CPU 可控、哪些是 Metal 排隊硬限制。

**新增診斷能力**(3 處計時點):
- `[cpu-chain]` 行:readback / plan / io / cb2 四段 CPU 純計算牆鐘(Run.swift)
- `chainWallNanos`:cb1 `waitForCompletion` 返回到 routedCB.commit 的完整牆鐘(決定性測量)
- `cb1WaitCallNanos`:waitForCompletion 呼叫本身的牆鐘

**乾淨窗實測**(r4,pool64-sync+tensorops+B4+w8,MTP off,256 tok,decode 19.8 tok/s):

```
[stage]      cb1=0.29s io=1.27s cb2=0.08s head=1.28s gpuWait=14.65s wall=17.58s
[cpu-chain]  readback=0.01s plan=0.04s io=1.27s cb2=0.08s total=1.40s
             chainWall=1.66s unaccounted=0.26s cb1WaitCall=9.33s ofWall=8%
[cb-latency] wait=9.33s gpu=3.34s wake=0.94s sched=5.15s other=0.00s ofWall=53%
[timeline]   gpuBusy=8.41s gpuIdle=4.49s gaps=24049 1-5ms:1057x=1.96s >5ms:32x=0.21s
[gpu-idle-after] sharedFFN=3.32s(7639x,0.4ms avg) phase1Hit=0.92s(601x,1.5ms avg)
```

**三個關鍵修正(推翻 §13.8 的初步結論)**:

1. **sched 5.15s 不是「調度空轉」— 大部分是正常 pipeline 重疊**。cb-latency 的 sched = cb1 commit→gpuStart,這段時間 GPU 在跑**前一層的 routedCB + sharedFFN**(gpuRoutedNanos 5.16s 就是證據)。cb1 排在 routed 之後是正確的流水線 — GPU 有活幹,不是空轉。先前把 sched 10.6s 當最大成本是**歸因錯誤**:那是 GPU 忙前一層,不是 CPU 提交慢。

2. **CPU 準備鏈真實只有 1.66s(chainWall),不是 8.6s**。`[cpu-chain]` 四段加總 1.40s,chainWall(含所有縫隙)也只有 1.66s — unaccounted 僅 0.26s(34us/層)。先前把 gpuIdle-after-sharedFFN 的 8.62s 全歸給 CPU 準備是錯的:該 gap 的大頭是 **wake 延遲 + Metal 排隊**,不是 CPU 在忙。

3. **真正的成本分布(每層 2.3ms = 19.8 tok/s 的倒數)**:
   - **GPU 執行 1.10ms/層**(attn 3.34 + routed ~2.2 + shared + head):Metal 硬成本,只剩 kernel 優化
   - **wake 0.94s = 123us/層**:GPU 完成→CPU 喚醒的 Metal 完成處理器延遲 — **部分可控**(polling/spin 取代 blocking wait 可回收一半以上)
   - **chainWall 1.66s = 218us/層**:CPU 準備鏈(io 167us 為主)— **完全 CPU 可控**(hit-only fetch 再簡化、arg buffer 重用)
   - **sched 5.15s**:主要是 GPU 忙前一層的正常重疊 — **不是損失**

**哪些 CPU 可控、哪些是硬限制**:

| 成分 | 規模 | 屬性 | 可回收 |
|---|---|---|---|
| GPU 執行(attn+routed+shared+head) | 8.41s(48%) | Metal 硬成本 | 僅 kernel 優化 |
| wake(GPU 完成→CPU 喚醒) | 0.94s(5%) | 半可控 | polling 可收 ~0.4-0.6s |
| chainWall CPU 準備鏈 | 1.66s(9%) | **CPU 可控** | io 1.27s 是最大單項,hit-only 可再簡化 |
| sched(cb1 排隊) | 5.15s(29%) | 正常 pipeline 重疊 | 非損失,勿追 |
| 其餘(head 後空轉等) | ~1.4s(8%) | 混合 | 小 |

**一句話**:GPU busy 65% 的天花板不是 sched — sched 是假的。真實剩餘空間 = **wake 0.94s + chainWall 1.66s ≈ 2.6s(≈15% decode)**,其中 CPU 可控的是 chainWall 的 io 1.27s 與 wake 的一半。**B4 之後 io 仍是每層最大的 CPU 單項(167us/層)**:hit-only fetch 仍付 cacheLock + 計數器 + 雙重 view 構建稅,值得第三輪簡化。

### 13.13 wake polling + io 拆解(2026-08-08)— 推翻 §13.12 前提、wake 是實槓桿

**動機**:用戶要求「hit-only fetch 第三輪簡化(砍 io 一半)+ wake polling」同時做。加了 fetch 分支計數器([cpu-chain] 新增 hitIo/missIo)後,用資料說話:

```
[cpu-chain] io=1.12s hitIo=0.01s missIo=1.10s   (r4 pool64-sync w8, 128 tok)
[expert-cache] decode req=30480 hit=29927 rate=98.2%
```

**推翻 §13.12 前提:io 1.27s 不是 hit-path 稅,是 miss pread**。hit-only sync fetch 已接近免費(hitIo=0.01s ≈ 3us/層),「第三輪簡化 hit-only fetch」無從砍起。io 的 99% 是 miss 讀取:553-707 個 miss 專家 × 3.2MB ÷ ~1.75GiB/s ≈ 1.1-1.3s。miss 分布分析(256 tok trace vs profile):**798 個不同非 pool 專家,top-40 只覆蓋 20.6% — 純長尾**,pool promotion 無效。

**miss 槓桿窮盡清單**:
- pool 64→80:記憶體牆(7.7GB sync commit,§13.10 已證)
- slots 96→112(LRU 32→48,lazy 記憶體):**實測崩壞** — hit 率掉到 62.1%、讀 78GiB、decode 9.5 vs 12.3 tok/s(pool 載入異常,勿用)
- pool promotion(熱 miss 交換冷 pool 成員):長尾分布下 top-40 僅覆蓋 20.6%,ROI 極低
- **剩餘 miss 是 SSD 頻寬下限,唯一出路是 r3/r2 更小 expert(§13.4 已定案)**

**wake polling — 實槓桿**:

實作:`TURBO_FIELDFARE_WAKE_POLL_US`(0=off,默認 off;prod 設 5000)。decode loop 的 cb1 wait 改為 `waitForCompletionPolling`:先 spin `cb.status`(sched_yield 間隔,不飢餓其他執行緒),deadline 內完成即省掉 waitUntilCompleted 的 semaphore 喚醒;超時才 fallback blocking。fused 是 off(split path),sharedCB 在 cb1 後跑 ~150us,wake(120us)延遲 CPU chain 開始 → 暴露 188us/層;polling 讓 chain 提早 ~100us/層開始,且**對負載噪聲免疫**(parked thread 喚醒在負載下膨脹,spinning 不受)。

**A/B 數據(2026-08-08,重開機後)**:

| 場景 | off | poll-5000 | 說明 |
|---|---|---|---|
| 4 輪交錯(MTP off,負載噪聲窗) | 11.6 median | **15.7 median** | **+35%** |
| 診斷窗(load ~2-3) | 8.3-17.2 飄 | 16.3-17.1 穩定 | polling 消除負載敏感 |
| spin 窗口 1500/5000/20000us | — | 16.3/17.1/16.5 | 5000 是甜蜜點(覆蓋 avg wait + miss 讀) |
| 完整 prod(MTP on,3 輪交錯,有效 off) | 12.13 | **13.89** | **+14.5%** |
| byte-identity | — | ✓(code+prose md5 一致) | 只改等待機制,不動資料面 |

**成本**:decode 執行緒 CPU 平均 43%→70%(+28%,單一執行緒 sched_yield spin)。換來的是**任何負載下穩定 ~16-17 tok/s** — 對這台跑 Doubao/WindowServer 的機器是實際勝利。

**prod 默認**:`TURBO_FIELDFARE_WAKE_POLL_US=5000`(run_prod.sh 用 `${VAR:-5000}` 尊重 caller override,opt-out=0 有效)。prefill/TTFT 的 wait 未接 polling(長 kernel 上 spin 浪費 CPU,已確認 TTFT 無稅)。code review 修正:`.error` 終態立即 fallthrough、`&*` 防 overflow、missIo `&+` 一致性。

**一句話**:§13.12 的「hit-only fetch 第三輪簡化」前提錯誤 — io 是 miss pread 不是 hit 稅(hitIo=0.01s);真正的新槓桿是 wake polling,在負載下 +35%、安靜窗中性、byte-identical,已設為 prod 默認。

### 13.14 MTP 全面定案為默認關閉(2026-08-08)— 三種位寬全負收益

**動機**:13.13 後用戶要求驗證 r4 上 MTP 是否仍正收益(先前接受率 72-88% 是 r4 數據)。若 r4 也負,就把 MTP 默認關掉。

**r4 交錯 A/B(256 tok,load ~2.3,3 輪)**:

| 位寬 | MTP-on(adaptive) | MTP-off | Δ |
|---|---|---|---|
| r2 | 17.1 | **23.5** | **−37%** |
| r3 | 16.7-17.0 | **18.6-19.2** | −12% |
| r4 | 15.3 | **19.7** | **−29%** |

**三種位寬 MTP 全是淨負收益**。根因(base 變快後 verify 不划算 + gate 校準盲點,見 13.13):adaptive gate 的 d=0 baseline 走 MTP loop 量測(含 loop 自身序列化開銷),低估真 base 速度 → 誤判 MTP 划算而保持 drafts。

**執行**:run_prod.sh 的 `MTP_MODEL` 默認改為**空**(MTP off),保留 `MTP_MODEL=/path` 顯式開啟。`${MTP_MODEL-}` 語義:UNSET 或空 = off;只有顯式非空才開。

**新默認端到端(r3)**:MTP-off **20.7 tok/s** vs 舊 MTP-on 17.7 — 默認配置直接 +17%,且 r3 也破 18 紀錄。override 驗證:MTP_MODEL=gemma-4-mtp-head 正常回 MTP-on 17.7(mtp=66/80, acc=81%)。

**生產默認更新(完整)**:
- r3 默認 + pool64-sync + tensor-ops + B4 + w8 + wake-poll 5000 + **MTP off** + trust-receipt
- = **20.7 tok/s**(r3)、23.5(r2)、19.7(r4,MTP-off)

**一句話**:MTP 在 base 變快後全面負收益(verify 82% 計算成本 > 接受率收益),默認關閉;r3 默認現在 20.7 tok/s,超越歷史上限 18。

### 13.15 r2(MTP-off)GPU 拆解與天花板(2026-08-08)— 24.1 tok/s = 76% of ceiling

**動機**:r2 MTP-off 23.5-24.1 tok/s 為目前最佳,用戶要求拆解剩餘縫隙並精算天花板。

**完整拆解(256 tok,decode 10.61s,load ~2.3)**:

```
[gpu] attn=3.25s routedMoE=2.38s sharedFFN=1.08s phase1Hit=0.14s head=1.17s busy=8.03s ofWall=58%
[cb-latency] wait=7.70s gpu=3.25s wake=0.82s sched=3.72s
[cpu-chain] io=0.76s hitIo=0.01s missIo=0.75s total=1.64s chainWall=1.11s
[timeline] gpuBusy=8.01s gpuIdle=2.59s  >5ms stalls: 1x(0.01s) — 全消失!
[gpu-idle-after] sharedFFN=1.74s(7650x,0.23ms avg) phase1Hit=0.60s(638x,0.94ms) head=0.17s routed=0.07s
```

**天花板精算**:
- GPU 每 token = 8.02s/256 = **31.3ms → GPU-only ceiling = 31.9 tok/s**
- 85% busy 實務上限 = **27.1 tok/s**
- 現在 24.1 = **76% of GPU ceiling**(r4 是 46%,r2 因權重小 GPU 時間縮短而大幅改善)
- 剩餘 2.59s idle = 10.1ms/token:sharedFFN-after 1.74s + phase1Hit-after 0.60s + head 0.17s

**每層預算(精確歸因)**:
- GPU 執行 1.04ms/層;CPU chain 252us/層(chainWall 1.11s + wake 0.82s);shared FFN GPU 141us/層
- **hit 層:chain ~150us ≈ shared 141us → 已近最佳**(med idle 0.13ms)
- **miss 層(784 請求 = 1.3%):chain 膨脹到 ~2ms → 全部 idle 來源**(0.60s phase1Hit-after + sharedFFN-after 大頭)
- w8 + r2 2.15MB 小權重已消滅 >5ms 長停頓(1x only)— 上一輪 miss 問題只剩短暴露

**下一個槓桿(排序)**:
1. **miss 讀取(0.75s missIo + 0.60s phase1Hit-after ≈ 1.3s,潛在 ~12%)**:最後未試的 lookahead-miss-prefetch — 在 cb1 GPU 窗口用下一層 router 預測,只讀**非 pool 的 miss**(預測命中率 57%,避開 pool 成員 → 零冗餘 I/O、零鎖競爭)。回收一半 ≈ decode 10.61→9.9s ≈ **26.5-27 tok/s**
2. chain 殘餘(0.23s unaccounted + plan/cb2 0.12s):微
3. attn 3.25s(最大 GPU 單項):128-tok 下 B3 phase-2 增益小

**一句話**:r2 已到 76% of ceiling(24.1 vs 31.9),hit 層已近最佳;剩餘 2.59s idle 幾乎全是 784 個 miss 讀取的短暴露(w8 已消滅長停頓)。下一個實槓桿 = lookahead 只預取非 pool miss,潛在 24.1 → 26.5-27 tok/s。

### 13.16 lookahead-miss-prefetch 證偽(2026-08-08)— 預測長尾不準,全面負收益

**動機**:§13.15 指出剩餘 idle 幾乎全是 miss 讀取(0.60s phase1Hit-after + missIo 0.75s),下一個槓桿候選 = lookahead-miss-prefetch(在 cb1 GPU 窗口用下一層 router 預測,只讀非 pool 的 miss)。

**實作**(TURBO_FIELDFARE_MISS_PREFETCH=1,默認 off):lookahead 預測在 readback(L) 產出後立即 dispatch 非 pool 預測專家,async 預取到 L+1 的 LRU,下層 plan 前 drain。V2 加「只預取本 run 已 miss 過」(known-misser 過濾)。

**V1 結果(r2)**:機制成功 — missIo 0.76→0.32s(−58%)、phase1Idle 0.60→0.22s(−63%) — 但 **tok/s 24.1→21.5(−11%)**。missPrefetch=4985 次(65% 層),預測長尾大多錯誤 → 讀了 ~7500 專家(vs 實際 784 miss)浪費 SSD。

**V2 結果(r2)**:known-misser 過濾後 dispatch 3274(仍 43% 層),但 **missIo 沒降(0.78s)、tok/s 仍 −7%**。根因數據:
```
[lookahead] predicted=59160 hit=40295 rate=68.1% prefetchRead=102 drain=0.05s
```
- lookahead 整體準確 68%(但被 pool 成員主導;長尾子集遠低)
- **prefetchRead=102**:3274 次 dispatch 只實際讀了 102 個專家 — 其餘候選 dispatch 時已 resident(plan 說不用讀)→ 純白付 dispatch + bg 執行緒 + cacheLock 開銷
- 真正 784 個 miss 是「首次長尾」或「被預測漏掉」— 預測命中不了

**r3/r4 確認**:r3 off 21.4 vs on 21.1(中性);r4 off 17.8 vs on 16.1(更糟,權重更大 drain 暴露更多)。

**結論**:miss-prefetch 全面負收益 — 預測對長尾的準確率撐不起 speculative read。代碼保留 env-gated off(可複現實驗),**prod 未接**。r2 剩餘 2.59s idle 的 miss 部分確認無 cheap 槓桿:pool 記憶體牆(§13.10)、slots112 崩壞(§13.13)、prefetch 負收益(本節)。**r2 現狀 = 24.1 tok/s = 76% of GPU ceiling(31.9) 就是這台機器+此分布的實際終點,除非架構級多 token 並行。**


## 13.17 B3 phase-2 ROI 評估（attention core 成本歸因，2026-08-08）

用 env 開關（`TURBO_FIELDFARE_SKIP_ATTN_CORE/SWA/FULL=1`，默認 off，prod 未接）跳過
attention core kernel（QKV / epilogue / OProj / router 照跑），量 cb1 GPU 時間 delta。

### 架構事實

- `D=2816`、16 Q heads / 8 KV heads（GQA 2:1）、30 層、5 層 full-attn（mask 位元 1）
- SWA 層：`headDim=256`，走 tiled split path（partial + combine 多 kernel + 中間 buffer）
- Full 層：`headDim=512`，走 tensor-ops 單 kernel（`512/16/2` geometry）

### 決定性測量（256 tok、乾淨窗）

| 指標 | core on | core skipped | Δ |
|---|---|---|---|
| `[gpu] attn` | 3.58s | 2.29s | **core = 1.29s = cb1 的 36%** |
| decode | 13.08s | 8.43s | 19.6 → **30.4 tok/s** |
| GPU busy | 9.15s | 6.77s | −2.38s |

### 關鍵發現

1. **FLOPs 佔比預期完全錯誤**：seqLen=128 時 core 只有 ~1-2M MACs/層 vs QKV 投影 ~24M
   （理論 ~5-10%），但實測 **佔 cb1 GPU 的 36%** — core 是 launch-bound（7680 次
   encode、每次 ~168us 對 1-2M MACs 是災難級低佔用），不是 compute-bound。
2. **放大係數 3.6x**：core 省 1.29s GPU → decode 省 4.65s。cb1 是串行鏈第一個環節，
   提早完成 → 每層 wake/chain 提早開始 → 全鏈縮短。routed/shared/head 也各快
   （GPU busy 額外 −1.1s）。
3. **128 tok 交錯歸因**（3 輪中位）：skip-SWA 省 0.59s、skip-FULL 省 0.99s、
   skip-both 省 3.20s（on=10.48s）。SWA 25 層每層 0.024s、FULL 5 層每層 0.20s —
   **full 層每層成本是 SWA 的 8 倍**（512 headDim + tensor kernel 仍低佔用）。

### ROI 結論：值得做

- phase-2 合理目標：把 core 的 launch-bound 轉成 compute-bound（SWA split path
  合併成單 pass tensor-ops + full kernel 佔用改善），core 砍半 → 省 ~0.65s GPU
- 套用 3.6x 放大 → **decode 13.08 → ~10.8s → 23.7 tok/s（+21%）**
- 保守估計（放大 2x）：+10-15% decode
- 這是 miss-prefetch 證偽後目前唯一的實質 decode 槓桿（wake-poll 已定案 +35%）

### 優先級建議

1. **SWA split path → 單 kernel tensor-ops**（25/30 層，主體工作，中等工程）
2. **full tensor-ops kernel 佔用檢查**（5/30 層但每層成本 8 倍，先查為什麼低效）
3. 不做：長上下文 chunked（128-256 tok 下 core 已佔 36%，chunk 對短上下文無增益）


## 13.18 full-attn tensor-ops kernel 低佔用診斷（B3 phase-2 第一步，2026-08-08）

### 問題

§13.17 顯示 full-attn 5 層每層 cost 是 SWA 25 層的 8 倍（0.20s vs 0.024s）。
本節找出 launch-bound 的具體機制。

### Kernel 結構事實（attention_prefill_full_tensorops_2d_validity_v2）

- **復用 prefill 的 MPP tensor kernel**，decode 時 `queryCount=1` → grid 只有
  `width:1 × height:numQHeads/8=2` = **2 個 threadgroup × 128 threads = 256 threads**
- `kPrefillTensorOpsOutputs=8`：每個 TG 處理 8 個 Q head（1 個 KV-head group）
- `execution_simdgroups<4>`：128 threads = 4 simdgroup 跑 matmul2d（QK、PV）
- 每 chunk 迭代 64 keys，seqLen=128 → 2 次 key loop 迭代
- **softmax 段序列化**：`if (lid < kPrefillTensorOpsOutputs)` — 每次 chunk 迭代只有
  8/128 threads 工作（其餘 120 threads 空等 barrier），8 個 thread 串行掃 64 keys

### 對照：SWA split path（256/16/8）

- partial pass：`numKVHeads(8) × numChunks(8) = 64 TG × 256 threads = 16,384 threads`
- combine pass：16 TG
- 並行度是 tensor kernel 的 **64 倍**

### 為什麼 full 層 8 倍貴（綜合證據）

| 維度 | tensor kernel (full 512/16/2) | split path (SWA 256/16/8) |
|---|---|---|
| threadgroups | **2** | 64 |
| threads | **256** | 16,384 |
| 每 TG 工作 | 8 Q heads × 512 dim | 1 KV head × 1 chunk |
| softmax 並行 | **8 threads 串行掃** | spread across TG |
| headDim | 512（理論 2× SWA） | 256 |
| 實測每層 | 0.20s | 0.024s |

FLOPs 只有 2 倍，但並行度差 64 倍 + softmax 序列化 → 實測 8 倍。這是
**launch-bound / 低佔用**，不是 compute-bound。kernel 是為 prefill（大量 query
tokens，grid = queryCount × 2）設計的，decode 單 token 時 MPP 優勢吃不到。

### 決定性實驗：tensor-ops on vs off（同窗 GPU 對照）

| 指標 | tensor ON | tensor OFF (split path) | Δ |
|---|---|---|---|
| attn GPU | 2.62s | **2.02s** | **−0.60s（−23%）** |
| GPU busy | 7.08s | **5.35s** | −1.73s |
| cb1Wait | 7.58s | **5.42s** | −2.16s |
| sched | 4.67s | **3.13s** | −1.54s |
| byte-identity | — | **IDENTICAL** | 零品質損失 |

full-attn 走 split path（32 TG × 256 = 8,192 threads）比 MPP tensor kernel
（256 threads）快 23% attn GPU。**不需要等 phase-2 重寫 — 直接關 tensor-ops
就是免費的修復**。

### 誠實的不確定性

decode 牆鐘的 3 組交錯 A/B 方向不穩（off 勝 17%、off 勝 9.6%、on 勝 15%）—
負載 2.7-4.1 噪聲把 ±10-15% 差異淹沒。GPU-side 證據（同窗對照）一致指向 off，
但最終 decode 定案需要乾淨窗。

### 附帶 bug

`decodeTensorOpsEnabled` 是 `environment["TURBO_FIELDFARE_ATTN_TENSOROPS"] != nil` —
註解宣稱「opt-out with =0」但 `=0` 仍是存在即開。應改 `== "1"`。


---

## §13.48 Qwen3.6-35B-A3B routing trace（2026-08-08 實測）

**方法**：streaming forward（真實權重：embedding + 全部 40 層注意力 + router；分片 22/26 已於 2026-08-08 補齊後重跑完整 40 層），
取 2 個 prompt（code 152 tok / prose 196 tok）收集每層 router 的 top-8 選擇。
**近似驗證**：shared-expert-only 前向 vs exact top-8 expert FFN 的 routing 一致率
layer0 100% → layer10 73%（§本節驗證腳本 /tmp/validate_approx.py），近似統計有效。

### 熱集集中度（256 experts, top-8/層）
| 指標 | code | prose |
|---|---|---|
| 使用到的 experts | 256/256 | 256/256 |
| top-64 覆蓋 | 40.0% | 43.7% |
| top-96 覆蓋 | 53.9% | 58.3% |
| top-128 覆蓋 | 66.0% | 70.2% |
| top-192 覆蓋 | 86.0% | 89.1% |
| per-layer top-64 覆蓋（中位）| 80% | 90% |

**關鍵結論：Qwen3.6 的 routing 遠比 Gemma4 平**——256 experts 全被使用、
全局 pool 覆蓋率低（pool64 僅 ~40-44%，vs Gemma4 pool64 95%+）。
且 **per-layer 熱集不同**（per-layer top-64 覆蓋 74-97%，但全局 top-64 pool 只有 40-44%）——
同一個全局 pool 對不同層各自的最佳熱集只能命中一半。

### decode 速度預估（503MB/token 未緩存、5.3GB/s、compute floor 26.3 tok/s）
| pool | hit | missMB/tok | 預估 tok/s |
|---|---|---|---|
| 64 | 40-44% | 283-302 | ~10.5-11 |
| 96 | 54-58% | 210-232 | ~12-13 |
| 128 | 66-70% | 150-171 | ~14-15 |
| 192 | 86-89% | 55-70 | ~19.5-20.7 |
| 256（全常駐）| 100% | 0 | ~26（compute 天花板）|

**白皮書 §1.2 的「503MB/token = Gemma4 的 0.65×」誤導**：原始未緩存 IO 確實較小，
但 pool 命中率遠低於 Gemma4（routing 更平 + 層間熱集不同），**端側實測速度反而更依賴池大小**。

### 對工程決策的影響
- **pool64 只夠 Gemma4**；Qwen3.6 需要 pool≥128 才到 14-15 tok/s
- **全常駐 256 experts 是質變**：僅 ~412MB int4（1.6MB×256），16GB 機完全可行，
  且 MTP verify 的 expert 並集問題也消失（全常駐 = 零 miss）→ **Qwen3.6 上 MTP 有機會轉正**
- 分片 22/26 下載完成後應重跑層 32-39 補全（腳本 /tmp/qwen36_route_trace.py）
- trace 腳本與分析在 /tmp/qwen36_route_trace.py、/tmp/analyze_routes.py、/tmp/validate_approx.py


---

## §13.49 Qwen3.6-35B-A3B 全常駐記憶體帳單（2026-08-08，int4 實測 shapes）

### 權重（int4，含全部 26 分片下載完成後的真實 tensor shapes）
| 項目 | 大小 |
|---|---|
| experts（256 × 30 層）| 12.08 GB |
| shared experts | 47 MB |
| full-attn dense（10 層）| 136 MB |
| DeltaNet dense（30 層）| 503 MB |
| router | 8 MB |
| embed（248,320×2048）| 254 MB |
| lm_head（未 tied）| 254 MB |
| **主模型小計** | **13.28 GB** |
| MTP head（fc + 1 層 decoder 含 256 experts）| 420 MB |
| **權重總計** | **13.70 GB** |

### Runtime
| 項目 | 大小 |
|---|---|
| KV cache（bf16，10 層 full-attn，2 KV heads×256）| 20 KB/token（8K = 168 MB）|
| DeltaNet state（32×128×128 fp32 + conv）| 2 MB（**非白皮書聲稱的 60MB**）|
| activations（~200 tok 峰值）| ~200 MB |

### 對照 16GB Mac
| 方案 | 權重 | 總計（4K ctx）| 16GB 可行？ |
|---|---|---|---|
| **int4 全常駐 + MTP** | 13.70 | 16.1 GB | ❌ 超標（可用 ~13-14GB）|
| int4 全常駐 − MTP | 13.28 | 15.6 GB | ⚠️ 邊緣（embed/lm_head 需流式）|
| int4 全常駐，embed/lm_head 流式 | 12.77 | 15.1 GB | ⚠️ 邊緣 |
| **int3 experts（同 r3 方案）+ MTP** | 10.7 | 13.1 GB | ✅ 可行 |
| **pool192 + 流式其餘 + MTP** | 9.7 | 12.2 GB | ✅ 可行（pool192 hit 85-87%）|
| int4 + MTP + KV int4 | 13.70 | 15.9 GB | ❌ 仍超標 |

### 結論
- 白皮書「13GB int4」低估（漏算 embed+lm_head 508MB 與 MTP head 420MB）；**真 int4 全常駐 = 13.7GB 權重 + runtime = 16.1GB，16GB Mac 裝不下**
- **全常駐 256 experts 單項只有 12.08GB**——真正的瓶頸是 embed/lm_head（508MB）+ MTP head（420MB）+ KV
- 三條可行路徑：**① int3 experts**（驗證過的 r3 方案，品質 gate 過）→ 13.1GB 含 MTP；**② pool192**（hit 85-87%）→ 12.2GB 含 MTP；**③ embed/lm_head 流式** → 15.1GB 邊緣
- DeltaNet state 僅 2MB（非 60MB）——**262K 上下文全常駐的記憶體優勢比白皮書宣稱的更強**

---

## §13.50 DeltaNet 層對 batched verify 的加速（2026-08-08 量化）

校準：Gemma4 session 實測 full-attn 層 0.20s / SWA 層 0.024s @(B=1, ctx=128)。
Qwen3.6 full-attn head_dim=256（=2× Gemma4 128 的 score 成本）、kv_heads=2（=1/4 Gemma4 的 KV 讀取）。
模型：score ∝ B×ctx×q_heads×head_dim；KV 讀 ∝ B×ctx×kv_heads×head_dim；DeltaNet ∝ B（2MB 固定 state）。

### B=3 verify 的 attention 成本（相對 gemma4 單步 = 1.0x）
| ctx | Gemma4 | Qwen3.6 | 誰便宜 |
|---|---|---|---|
| 128 | 3.0x | 5.5x | Gemma4（短 ctx）|
| 2K | 16x | 5.1x | **Qwen（0.32×）** |
| 8K | 24x | 18x | **Qwen（0.76×）** |
| 32K | 56x | 70x | Gemma4 略 |
| 131K | 184x | 280x | Gemma4 |

### 關鍵發現：是 crossover，不是單向優勢
- **短 ctx（<2K）Qwen3.6 反而貴 1.8×**——10 層 full-attn 的 head_dim=256 太貴（每層 = Gemma4 的 2×），DeltaNet 省的 ctx 依賴在短 ctx 沒有意義
- **中 ctx（2K-8K）Qwen 顯著便宜**——30/40 層停止隨 ctx 增長，只有 10 層 full-attn 在漲
- **長 ctx（32K+）Gemma4 又反超**——Gemma4 有 SWA window（2048 封頂，25 層固定）+ 僅 5 層 full 無封頂；Qwen 有 10 層 full 無封頂且 head_dim 256，平方項 2× 較重

### 實際意義（M4 16GB）
1. **DeltaNet 的優勢 = ctx 縮放行為，不是每 token 成本**——262K 下 30 層永遠不漲，才是架構真正的長上下文武器
2. **verify 的每 token 成本主要由 MoE 權重讀取決定**（503MB vs Gemma4 768MB = 0.65×）——attn 在 4K-32K 典型區間只差 ±25%
3. **full-attn 10 層是長上下文牆**：KV 20KB/token × 262K = 5.1GB（§13.49）——若要跑長上下文，KV 量化或 full-attn 稀疏化才是下一步

### 結論
DeltaNet 對 batched verify 的加速**不是「每 token 便宜 2-3×」**（短 ctx 反而貴），而是**「30/40 層的成本隨 ctx 完全凍結」**——在 2K-32K 的實用區間內 Qwen3.6 verify 比 Gemma4 便宜 25-68%，長上下文（262K）時 30 層固定成本的價值才完全顯現。

---

## §13.51 B=3 batched verify 的權重讀取攤薄（Metal kernel 層，2026-08-08 量化）

### Kernel 事實
phase-1 MoE kernel（`moe_phase1_gate_up_act_u16load`）：`rowg = tg_idx*8 + sg_idx`，
`slot = rowg/F`——**同一 expert blob 由多個 threadgroup 讀取（不同 f），batch 內相鄰 token
選到同一 expert 時 blob 走 L2 重用**（M4 L2 16-24MB，r4 blob 3.2MB / qwen blob 1.6MB）。

### 並集實測（Qwen3.6 40 層 trace，top-8/256）
| B | union 中位 | union/8 | 重疊率 |
|---|---|---|---|
| 2 | 15/16 | 1.88 | 12% |
| 3 | 21/24 | 2.62 | 12.5% |
| 4 | 27/32 | 3.38 | 16% |
| 5 | 31/40 | 3.88 | 22% |

**Qwen3.6 top-8 並集幾乎線性增長**（B=3 重疊僅 ~12%）——「8 experts 載一次餵 3 token」的
假設在 Qwen3.6 上不成立（相鄰 token 的 top-8 幾乎不重疊）。

### 每 token MoE 權重讀取（並集讀一次，L2 重用）
| 模型 | 單 token | B=3 每 token | 攤薄 |
|---|---|---|---|
| Gemma4 r4（8×3.2MB）| 25.6 MB | 21.8 MB（union 85%）| 0.85× |
| Qwen3.6（8×1.6MB）| 12.8 MB | 11.2 MB（union 21）| 0.88× |

### 真正的攤薄在 dense 層，不在 MoE
- **dense 權重（attention + shared expert + norms）每層只讀一次餵 B tokens**：
  Qwen3.6 dense 40 層 ≈ 904MB → B=3 每 token 301MB = **3× 攤薄**
- **MoE 權重攤薄只有 0.85-0.88×**（並集重疊太少）
- 合計：單 token 總權重讀取 ≈ 1289MB；B=3 batch ≈ 1845MB → **每 token 615MB（1.4× 單 token）**
  → batched verify 的權重讀取沒有「攤薄到小於單 token」，只是「從 3× 變成 1.4×」

### 結論（修正「載一次餵 3 token」的直覺）
1. **權重讀取攤薄是假的**（Qwen3.6 top-8 並集 88% 線性）——batched verify 的權重流量
   是單 token 的 ~1.4×/token，不是 1/3
2. **batched verify 的真正紅利 = 固定成本攤薄**：CB 數（8808×330μs）、kernel launch、
   sched/idle gap、CPU→GPU sync——一次付給 3 tokens，這才是 B4 pipeline 的實質
3. **dense 層是免費午餐**：904MB 權重 B 個 token 共享，全常駐下這個優勢完整兌現
4. 對 Qwen3.6 的意義：verify 的權重讀取成本 = 11.2MB×30 MoE + 301MB dense ≈ 637MB/3 tokens
   ≈ 212MB/token——**遠小於 503MB 未緩存單 token**，因為熱集/全常駐時 L2 命中吃掉 MoE 大半
