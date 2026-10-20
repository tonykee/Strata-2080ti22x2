# 部署说明（2× RTX 2080 Ti 22G · **v0.1.45**）

> 本文档描述 **`~/strata` 主部署**的完整状态：版本、配置、启动、编译、实测、回退。
> **逐项实测账本见 [`OPT-2026-10.md`](OPT-2026-10.md)**（§9.x 最新，含每次 A/B 的数字与结论）。
> 旧版本（v0.1.39）的历史与 A/B 见 §6/§7、[`V0139-MIGRATION.md`](V0139-MIGRATION.md)、[`REPORT-v0143-2080ti.md`](REPORT-v0143-2080ti.md)。

## 0. 一句话（2026-10-16 迁移后）

- **引擎**：上游 **v0.1.45**（2026-10-17 升级）+ **pw 补丁**（`--pipeline-windows` 与 `--batch` 共存）。
  #1821 与 #743 已被上游吸收 ✓，**不再需要本地移植** ✓。
- **分支**：`v0.1.45-2080ti` = 上游 **`v0.1.45` 标签** + pw + 本地文档；**`main` 与 GitHub 都指这里**。
  旧主干完整存于 `backup-v0143-main-20261018`（v0.1.43 线）与 `backup-v0139-main-20261016`（v0.1.39 线）。
- **env 三项，全部有本机实测依据**：`STRATA_BF16_TC=1`（预填 **+22.6%**）、`STRATA_SELECT_SIMT=1`（预填 **+22.4%**）、
  **`STRATA_SM75_PERF=1`**（**上游 0.1.45 的 Turing 内核系列** ✓ 解码 **+4.9%**、预填 **+6.9%**，见 §5 与 §13）。
  （`STRATA_GR_V3` 已于 2026-10-16 **删除**：Turing 上被引擎自己拒绝、是空操作，见 `OPT` §9.19-§9.21。）
- **主部署模型**：`sc117-abliterated-iq3_s`（IQ3_S 原生 pack）；另有 **Swift**（`start_iqs3_swift.sh`）。
  （原 **qwen** / **Q2_0** / **Swift-IQ3_XXS** 三个部署已于 2026-10-16 **全部清除** ✓：
  模型 `~/models/qwen-IQ3_S`（78 GB）、各自的配置 / 启动脚本 / pack 一并删除 ✓ ——
  只留 **117** 与 **Swift-IQ3_S** 两套 ✓。详见 [`IQ3_S-NOTES.md`](IQ3_S-NOTES.md)。）
- **硬件**：2× RTX 2080 Ti 22G（Turing **sm_75**）、93 GiB RAM、SATA SSD、CUDA 12.8 / gcc-13。
- **服务**：`./start_iq3s_117.sh`，端口 **8000**，API key `llama_local`。
- **回退**：见 §12（旧分支 + 旧引擎 + 配置备份，三条都在）。

## 1. 硬件与系统

| 项 | 值 |
|---|---|
| GPU | 2× NVIDIA RTX 2080 Ti 22 GB（compute capability **7.5**） |
| GPU 互连 | **NVLink NV2**（`topo -m` = NV2；`nvlink -s` Link0/1 各 25.781 GB/s）。**引擎的层切分交接仍走 pinned host（~3.3 GB/s），未用 NVLink** |
| GPU 功耗上限 | **160 W（有意限功率：散热不好）**；实测负载中只跑到 ~110–128 W，低于上限、不是瓶颈。默认 250 W / 最大 280 W，**不要调高** |
| RAM | **93 GiB**（2026-10-08 从 61 GiB 升级；4 条混插：2×16 Juhor + 2×32 Cuso，其中一条 2400 2R Hynix → 全通道按 2400 跑，带宽实测 ~32 GB/s） |
| 系统盘 | WDC WD Blue SATA SSD（~547 MB/s） |
| CUDA / 编译器 | CUDA 12.8 / gcc-13 |
| 模型 | `~/models/sc117-iq3_s`（主，80 GB）／`~/models/swift-IQ3_S`（备，79 GB）；视觉共用 `~/models/mmproj` |
| pack | `~/Strata-data/packs/sc117_iq3_s`（主）／`swift-iq3_s`（备）|

## 2. 启动 / 停止

```sh
cd ~/strata
./start_iq3s_117.sh                 # 主部署（sc117），端口 8000
./start_iqs3_swift.sh               # 备部署（Swift），端口 8000
# 验证
curl -s -o /dev/null -w '%{http_code}\n' -H 'Authorization: Bearer llama_local' \
  http://127.0.0.1:8000/v1/models   # 期望 200（首次约 2 分钟）
# 停止
pkill -TERM -f '[s]trata/serve/server.py'; sleep 2; pkill -9 -f '[s]trata/serve/server.py'
pkill -9 -f '[s]trata/engine/strata '; pkill -9 -f '[s]trata/engine/strata-vision'
```

## 3. 配置文件

> 三个配置都被 `.gitignore` 忽略（含绝对路径），需单独维护。
> **主部署** = `strata-sc117-iq3_s.json`（`./start_iq3s_117.sh`）；备部署 = `strata-swift-iq3_s.json`（`./start_iqs3_swift.sh`）。
> 下面以 **sc117** 为例（Swift 只是 pack/native/mtp 路径不同，**env 完全一样**）：

```jsonc
{
  "exe": "/home/likan/strata/engine/strata",
  "args": [
    "--pack", "/home/likan/Strata-data/packs/sc117_iq3_s",
    "--native", "/home/likan/models/sc117-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "/home/likan/models/sc117-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", "/home/likan/strata/data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--spec", "4", "--spec-min-p", "0.65",
    "--mtp", "/home/likan/models/sc117-iq3_s/strata/rt",
    "--max-context", "262144",
    "--kv", "int8", "--kv-resident", "32768",
    "--vision",
    "--vram-reserve-mib", "700",
    "--trim-stage-weights",
    "--suffix-draft", "8",
    "--conversation-cache-mib", "16384",  // 会话停车（parking）：2026-10-11 由 8192 调大，见 §7
    "--conversation-cache-slots", "4",
    "--batch", "2", "--batch-groups", "2", // 两个 agent 真并行
    "--pipeline-windows", "2"              // pw 补丁：与 --batch 共存（单请求 +17%），见 §7
  ],
  "layer_split": "24",
  "gpu": [0, 1],
  "port": 8000,
  "api_key": "llama_local",
  "env": {
    "STRATA_BF16_TC": "1",           // ★ 必需，见 §5（预填 +22.6%）
    "STRATA_SELECT_SIMT": "1",       // ★ 必需，见 §5（预填 +22.4%）
    "STRATA_SM75_PERF": "1"          // ★ v0.1.45 的 Turing 内核系列，见 §5（解码 +4.9%、预填 +6.9%）
  }
}
```

## 4. 编译（从源码）

v0.1.45 的布局是 **`third_party/ggml`**（只有版本号）→ CMake 会**自己拉取** pin 的 llama.cpp（配置阶段约 90 s，
需要网络 ✓）。**不要**传 `-DSTRATA_GGML_DIR`（那是 v0.1.43 布局的用法 ✗）。

```sh
cd ~/strata
rm -rf build
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DSTRATA_GGML_DIR=/home/likan/strata/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
cmake --build build --target strata -j8
cp -f build/strata engine/strata
```

> 实测：**153/153** 目标、0 error（约 40 分钟，`-j8`）；产物 md5 `b24708ac8e39`（2026-10-18，v0.1.45）。

> 视觉编码器 `engine/strata-vision` 是独立 CMake 工程（`tools/vision`），沿用旧部署的即可；
> 需要时按 `MERGE-NOTES.md §6` 重编。

## 5. ★ 关键：sm_75 的两个 env（本机实测，缺一掉 ~22%）

两个都在 `env` 里，**都做过 A/B**（两臂各 2 样本，详见 [`OPT-2026-10.md`](OPT-2026-10.md) §9.18）：

| env | 作用 | 本机实测（100K 预填）|
|---|---|---|
| **`STRATA_BF16_TC=1`** | BF16 投影走 **FP16 张量核**（Turing 没有 BF16 张量核）| 1961 vs 1600 → **+22.6%** |
| **`STRATA_SELECT_SIMT=1`** | 块分数用**分块 FP32** 核，而不是默认的 warp 核 | 1603 → 1963 → **+22.4%** |

- `BF16_TC`：**会改变数值**（FP16 舍入；上游文档记 2080 Ti prompt +15~18%，本机量到 +22.6%）；对解码无影响。
- `SELECT_SIMT`：**v0.1.43 起默认走 warp 核**（上游文档称其为"pre-sm_80 的默认"），长 prompt 上慢约 **7 倍**
  （`block_scores` 内核 21.05 s → 0.03 s）→ 不设它，100K 预填会掉到 1600 上下。
- **`STRATA_SM75_PERF=1`（v0.1.45 新增 ★）**：打开上游为 **sm_75 写的整套内核**
  （PR #2069-#2072 / #2081 / #2091-#2093 / #2145-#2146 / #2220-#2223 / #1821 ✓）——
  **作者在 RTX 2080 Ti 上实测**，源码声称**逐位 bitwise** ✓，上游团队**没有 sm_75 卡** ✗ 所以默认关闭 ✓。
  本机四臂对照实测（见 §13 与 `OPT` §9.24）：**解码 +4.9%** ✓、**100K 预填 +6.9%** ✓。
- 三者叠加正是上游文档推荐的**最优组合**（见 `docs/OLDER_GPUS.md` 的 Quadro RTX 8000 实测表）。
- 另有 `STRATA_GR_V3` **已于 2026-10-16 删除**：Turing 上被引擎自己拒绝（`gr_parity` 非确定，#375），
  是空操作，见 `OPT` §9.19-§9.21。

## 6. 实测性能（本机 · **v0.1.39 时代，留作历史对照**）

> ⚠️ 下面是 **v0.1.39 + #910** 时期的数字。**当前 v0.1.43 的实测见 §13 与 [`OPT-2026-10.md`](OPT-2026-10.md) §9.x。**

`--batch 2` 或单并发（fresh，**multi 词表**，`STRATA_BF16_TC=1`）：

| 项 | 本部署（+#910） | 旧版（仅 #742/#743） |
|---|---:|---:|
| prefill 12.5K | ~1078 tok/s | ~1089 |
| **prefill 192.8K** | **~1773 tok/s** | ~1799 |
| **decode 中文（batch2）** | **~73.7 tok/s** | ~64.3 |
| **decode 中文（单并发）** | **~73.7 tok/s** | ~63.2 |
| decode 英文 | ~67 tok/s | ~67 |

- **长 prompt prefill 保持 #742/#743 的水平**：192.8K ~1770–1850 tok/s，读入 158 s → ~105 s（**+55%**），
  靠 #742（FP32 tiled block scores）+ #743（coalesced streaming top-k）。
- **decode 因 #910 提升 ~15–16%**：`54309f1`（无 #910）~63–64 → `982b651`（+#910）~73.7。
  - **前提是 multi 词表**（接受率高）。en 词表下中文接受率被压到 ~48%，看不到 #910 的收益——见 [`MTP-VOCAB.md`](MTP-VOCAB.md)。
  - `--batch 2` 下 `--pipeline-windows` 被引擎禁用，但仍到 ~73.7（batch 重叠）；单并发开 #910 也到 ~73.7。

## 7. 并发（2 用户）

- `--batch 2 --batch-groups 2`：两个会话**真并行 decode**（每个 verify window 各带一个 token），
  **入场（读 prompt）仍是一次一个**（另一个 slot 暂停）——这是引擎的单序列 prefill 设计。
- **`--pipeline-windows 2` 已与 `--batch` 共存**（本地补丁 `cea2d995`，见 [`LOCAL-DELTAS.md`](LOCAL-DELTAS.md) §1）：
  上游硬互斥（`not with --batch slots`），删掉该 guard 后单请求回到 solo 路径走流水线。实测（引擎自报 tok/s）：

  | | 单请求 | 双并发（每人 / 合计） |
  |---|---:|---:|
  | 改前（batch 2） | 58.8 | 32.0 / 31.9 → **63.8** |
  | **补丁后（batch 2 + pw 2）** | **68.8（+17%）** | 33.2 / 32.9 → **65.8** |

  prefill 不变（1274.6 → 1281.9 tok/s）。加压：solo↔双并发交替 5 轮无错、长上下文 100K + 停车正常。
- `--conversation-cache-mib 16384`（parking）：会话切走时快照到 RAM、回来时恢复，**后续轮次不重读历史**。
  实测（2026-10-11，本机 93 GiB RAM）：100K 会话首轮 100092 tok / 53.1s（1884 tok/s），
  追问 `100087 reused + 38 read` **1.7s**（整段重读要 ~52s）。
- **停车快照 ≈ 790 MiB + 15.0 KB/token**（实测：100K → 2.20 GiB，200K → 3.63 GiB，255K → 4.41 GiB）。
- **双并发的实测峰值 ≈ 8.8 GiB**（2 个 255K 会话各停 4.41 GiB，`evictions=0`，追问 100% 复用）→ 16384 有约 85% 余量。
- **池子必须装得下工作集，否则驱逐 → 整段重读**（实测，池 8 GiB）：
  4 个 100K 会话（4×2.36 = 9.4 GiB）逐个回访 → **#2/#4 `0 reused + 1001xx read`（各 54s）**，只有 #1/#3 复用。
  同一测试换池 24 GiB → 4 条全部 `1000xx reused + 38 read`（1.1–1.9s），**`evictions=0`**。
  → 2026-10-11 由 8192 调大到 **16384**（按双并发峰值 8.8 GiB + 余量；`--conversation-cache-slots 4` 全满需 17.6 GiB，
  只在 4 个满长会话同时停车时才需要）。该参数是**预算不是预分配**（不用不占 RAM），
  `--conversation-cache-min-free-mib 2560` 兜底（RAM 不够时 skip parking，不会 OOM）。
- **已结案（原“待 96 GB 后复测”那两条）**：93 GiB 下 parking 稳定生效；2×200K 并发回访复用率 100%，
  不再出现 v0.1.39 的整段重读（2×100K 总 wall 207s）。
- **双并发缺 MTP：已移植上游 `--batch-mtp` 并实测 → 本机不划算，未采纳**（分支 `port-batch-mtp`）。
  实测（本机，双并发 2×512 token 合计）：`--batch-groups 2` 无 mtp **63.8**；`--batch-groups 1` 无 mtp **52.4**；
  `--batch-groups 1` + `--batch-mtp` **57.3**。即 batch-mtp 本身有效（**+9.4%**，草稿接受率 72.4%），
  但它**强制 `--batch-groups 1`**（两阶段串行），而分组流水线在本机值 **+22%**（52.4→63.8）→ 合起来仍比现状低 10%。
  移植后引擎在当前配置下中性（64.6 vs 63.8）。详见 [`OPT-2026-10.md`](OPT-2026-10.md) §9.6。
- **注意**：`--adapt-async` 不能和 `--batch` 共存（源码硬互斥，且需驻留 RAM 模式，而该模式**不支持层切分**）——
  跟内存无关，加内存也不会解锁。

## 8. 与旧部署的差异 / 回退（**v0.1.38 → v0.1.39 的历史对照**）

> ⚠️ 本节讲的是 v0.1.38→v0.1.39 那次升级。**当前（v0.1.43）的回退见 §12。**

| | 旧（v0.1.38 + cherry-pick） | 本部署（v0.1.39 + #742/#743） |
|---|---|---|
| 每卡只加载本层 dense | `--stage-weights`（本地补丁） | `--trim-stage-weights`（上游 #639） |
| 多并发 | #559 cherry-pick | 上游 #465/#559 重做版 |
| decode | #646 cherry-pick | 上游 #646 重做版 |
| prefill 长 prompt | #575 cherry-pick | **#742/#743**（更强） |
| sm_75 BF16→FP16 | 本地 #593（默认开） | 需 `STRATA_BF16_TC=1` |
| decode 双卡流水线 | — | **#910 栈**（`--pipeline-windows` / `--adapt-async` + `STRATA_*`） |
| MTP 草稿词表 | 英文 40525 | **多语言 106299**（见 `MTP-VOCAB.md`） |

- **回退分支**：`batch-559-646`（v0.1.38+补丁）或 `port-v0134`。
- **回退引擎**：~~`engine-bak-opt/strata-v0138-batch559`~~（该备份已于 2026-10-16 删除 ✗）；**当前回退见 §12** ✓。
- **GitHub**：`main` = 本版本；旧版在 `v0.1.38-legacy`。

## 9. 非 git 资源（换目录/重装时要单独带）

| 路径 | 说明 |
|---|---|
| `engine/strata`、`engine/strata-vision` | 本地编译的引擎（sm_75） |
| `build/`、`build-vision/` | CMake 构建目录 |
| `strata-sc117-iq3_s.json`、`strata-swift-iq3_s.json`（+ sc117 的 `-pleram` 变体）| 配置（gitignore；见 §3）|
| `mtp-rt/` | MTP 目录（**multi 草稿词表** + 软链到 `Strata-data/mtp/rt`）；en 备份 `draft_vocab.bin.en.bak`。**⚠️ `Strata-data/mtp/rt/`（787 MB）是运行必需的** ✓（Swift 经软链读它 ✓）—— **不要删** ✗ |
| `data/expert-profile.bin` | 专家 profile |
| `.bench/` | 基准脚本/请求/结果 |
| `engine-bak-0139/` | **唯一的引擎回退备份** ✓（其余 `engine-bak-*` 已于 2026-10-16 删除 ✗）|

## 10. 上游同步

- **本分支 = 上游 `v0.1.45` 标签 + pw 补丁 + 本地文档** → **与上游同源**，
  近期上游 PR 能**直接 cherry-pick** ✓；#1821 与 #743 已在上游 ✓（**不再需要本地移植** ✓）。
- **升级到 v0.1.45 时只带了 pw 一个补丁** ✓（`--pipeline-windows 2` 与 `--batch 2` 共存 ✗ 上游仍拒绝 ✓）。
- 更早的主干（v0.1.43 / v0.1.39）分别存于 `backup-v0143-main-20261018` 与 `backup-v0139-main-20261016`。
- 同步上游新版本：`git fetch origin` 后 `git rebase <新标签>`，`--skip` 掉已在上游的提交。
  详细步骤见 [`MERGE-NOTES.md`](MERGE-NOTES.md) 与 [`LOCAL-DELTAS.md`](LOCAL-DELTAS.md)。

## 11. PCIe 链路（2026-10-16 排查，**重要**）

**症状**：预填从 1500–1740 掉到 470–700（长上下文 −40%），引擎/磁盘/CPU/GPU 时钟全部实测排除后，
发现**引擎自探的 PCIe 带宽**从 **6.7 GB/s 掉到 1.7 GB/s**（Gen3 → Gen1）。

**根因**：`/etc/systemd/system/gpu-pcie-gen3-fix.service`（本意"强制 Gen3"）在 **2026-10-14 12:07** 那次运行时
把链路**锁在了 Gen1** —— 它的脚本先设 Gen3，再置 `SpeedDis`（Link Control 2 bit 5），而 `SpeedDis` **禁止链路
自动变速**；若置位瞬间链路正处于空闲的 Gen1，就永久卡在 Gen1（每次开机都有这个竞态）。

**修复**：`sudo systemctl disable --now gpu-pcie-gen3-fix.service`（一次性 ✓，之后开机不会再锁 ✓）
→ 链路回到 Gen3 ✓ → 预填恢复 **1891.5（100K）/ 1876.8（255K）** ✓

**复查方法**（不需要 root）：看引擎启动日志的 `PCIe probe:` 行，正常应为 **6.7 GB/s**；
若读到 **1.7 GB/s** 就是坏链路。空闲时 `nvidia-smi` 报的 `pcie.link.gen.current=1` 是 ASPM 正常行为，**不能**用它判断。

## 12. 回退（2026-10-16 迁移后）

| 要回退什么 | 怎么做 |
|---|---|
| **整个引擎回 v0.1.43** | `git checkout backup-v0143-main-20261018` → 按 §4 重建 → `cp -f build/strata engine/strata` |
| **只换回 v0.1.43 引擎** | `cp -f engine-bak-v0143/strata engine/strata`（md5 `46f6434fefa4`）|
| **只换回 v0.1.39 引擎** | `cp -f engine-bak-0139/strata engine/strata`（md5 `aa34de41`）|
| **配置** | 每份都有备份：`.bak-v0139` / `.bak-pre-v0143` / `.bak-pre-grv3del` / `.bak-pre-park2` / `.bak-pre-slots2` / **`.bak-pre-sm75`** |

- **GitHub**：`main` 与 `v0.1.45-2080ti` 同指主干；旧主干在 `backup-v0143-main-20261018`（v0.1.43）与 `backup-v0139-main-20261016`（v0.1.39）。
- **引擎二进制**：`engine/strata` = 主树在 `v0.1.45-2080ti` 上的构建；
  回退用旧引擎在 **`engine-bak-0139/strata`（唯一保留的引擎备份 ✓）**。
- **已清理**（2026-10-16）：`engine-bak-0131/0134/0138/opt/pr593575/pre-1741/pre-pwguard/test`（756 MB）、
  qwen / Q2_0 / Swift-IQ3_XXS 的配置·脚本·pack、`mtp-multi/`。
  ⚠️ 历史文档（`MERGE-NOTES.md` / `PREFILL-PORT.md` / `V0139-MIGRATION.md` / `LOCAL-DELTAS.md`）里
  提到的 `engine-bak-0131`、`engine-bak-opt` 等路径**已不存在** ✗。

## 13. 调优账本（全部本机实测）

| 项 | 实测 | 状态 |
|---|---|---|
| `STRATA_BF16_TC=1` | 预填 **+22.6%** | 已开 ✓ |
| `STRATA_SELECT_SIMT=1` | 预填 **+22.4%** | 已开 ✓ |
| PCIe 链路修复（§11）| 预填 **4.0×**（470→1891）| 已修 ✓ |
| 迁移到 v0.1.43 | 解码 **+5.1~10.1%**；预填持平/略好 | 已迁 ✓ |
| Swift 迁移 | 解码 **+2.4%**；预填持平 | 已迁 ✓ |
| `--pipeline-windows 2`（pw 补丁）| 单请求 **+17%** | 已用 ✓ |
| **升级到 v0.1.45** | 解码 **+6.0%** ✓（65.5 → 69.4，两轮顺序一致 ✓）| 已升 ✓ |
| **`STRATA_SM75_PERF=1`**（0.1.45 的 Turing 系列）| 解码 **+4.9%** ✓、预填 **+6.9%** ✓ | 已开 ✓ |
| ↳ 合计（v0.1.43 → v0.1.45+SM75）| 解码 **+11.1%** ✓、预填 **+4.7%** ✓ | ✓ |
| **扩到 384K（yarn 1.5）** | 解码 **无可测代价** ✓（70.8 vs 71.1，四臂对照 ✓）| 已加 2 个配置 ✓ |
| **512K（yarn 2.0）→ 日常** ✓ | 400K 预填 **1894 tok/s** ✓、needle 命中 ✓、停车恢复 2.8 s ✓；试用期 **313 请求 / 解码均值 70.3 tok/s** ✓（`LOW` 2 次、`stall` 3 次、驱逐 32 次 ✗ 都很轻）| **已定为日常** ✓（`./start_iqs3_swift.sh` 现在指向 512K）|
| **`--prompt-cache-every 8192`** | 10 万 prompt **秒回**（100084/100116 reused ✓）| 已用 ✓ |
| **`--conversation-cache-slots 2`** | 与 `--batch 2` 匹配 ✓，驱逐 70+ → 1~2 ✓ | 已用 ✓ |
| `STRATA_GR_V3` | **0**（Turing 上引擎自己拒绝）| **已删** ✓ |
| `#2689`（上游声称 decode +15%）| **0**（前提是单卡）| 未采纳 ✓ |
| nsys 内核扫描 | **0 个新提升**（纯诊断）| 已完成 ✓ |

> 逐项数字、方法与结论见 [`OPT-2026-10.md`](OPT-2026-10.md) §9.x。
> 当前 GPU 利用率 **79% / 88%**、功耗 **137 / 140 W**（上限 160 W，**不动**）→ 这台机器基本吃满了。
