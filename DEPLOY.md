# 部署说明（2× RTX 2080 Ti 22G · v0.1.39 + #742/#743 + #910）

> 本文档描述 **`~/strata` 主部署**的完整状态：版本、配置、启动、编译、实测、回退。
> 迁移背景与 A/B 见 [`V0139-MIGRATION.md`](V0139-MIGRATION.md)。

## 0. 一句话

- **引擎**：上游 **v0.1.39**（`6f32ec0`）+ **#742 / #743**（Turing 长 prompt prefill）+ **#910 栈**（双卡 decode 流水线等）。
- **分支**：`v0.1.39-2080ti`（= `origin/main` + #742 + #743 + #910 + 本地文档）。
- **MTP 词表**：`mtp-rt/draft_vocab.bin` = **多语言（106299）**，不是英文（40525）——★ 中文快慢的关键，见 [`MTP-VOCAB.md`](MTP-VOCAB.md)。
- **模型**：Swift 1.5 IQ3_XXS（GSQ-RCO，原生 pack）。
- **硬件**：2× RTX 2080 Ti 22G（Turing **sm_75**）、61 GiB RAM、SATA SSD、CUDA 12.8 / gcc-13。
- **服务**：`./start_iq3.sh`，端口 **8000**，API key `llama_local`。

## 1. 硬件与系统

| 项 | 值 |
|---|---|
| GPU | 2× NVIDIA RTX 2080 Ti 22 GB（compute capability **7.5**） |
| GPU 互连 | **NVLink NV2**（`topo -m` = NV2；`nvlink -s` Link0/1 各 25.781 GB/s）。**引擎的层切分交接仍走 pinned host（~3.3 GB/s），未用 NVLink** |
| GPU 功耗上限 | **160 W（有意限功率：散热不好）**；实测负载中只跑到 ~110–128 W，低于上限、不是瓶颈。默认 250 W / 最大 280 W，**不要调高** |
| RAM | **93 GiB**（2026-10-08 从 61 GiB 升级；4 条混插：2×16 Juhor + 2×32 Cuso，其中一条 2400 2R Hynix → 全通道按 2400 跑，带宽实测 ~32 GB/s） |
| 系统盘 | WDC WD Blue SATA SSD（~547 MB/s） |
| CUDA / 编译器 | CUDA 12.8 / gcc-13 |
| 模型 | `/home/likan/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` |
| pack | `/home/likan/Strata-data/packs/swift-iq3_xxs` |

## 2. 启动 / 停止

```sh
cd ~/strata
./start_iq3.sh                      # 端口 8000
# 验证
curl -s -o /dev/null -w '%{http_code}\n' -H 'Authorization: Bearer llama_local' \
  http://127.0.0.1:8000/v1/models   # 期望 200（首次约 2 分钟）
# 停止
pkill -TERM -f '[s]trata/serve/server.py'; sleep 2; pkill -9 -f '[s]trata/serve/server.py'
pkill -9 -f '[s]trata/engine/strata '; pkill -9 -f '[s]trata/engine/strata-vision'
```

## 3. 配置文件 `strata-swift-iq3_xxs.json`

> 该文件被 `.gitignore` 忽略（含绝对路径），需单独维护。关键参数：

```jsonc
{
  "exe": "/home/likan/strata/engine/strata",
  "args": [
    "--pack", "/home/likan/Strata-data/packs/swift-iq3_xxs",
    "--native", "/home/likan/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "/home/likan/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "--expert-profile", "/home/likan/strata/data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--spec", "4", "--spec-min-p", "0.65",
    "--mtp", "/home/likan/strata/mtp-rt",
    "--max-context", "262144",
    "--kv", "int8", "--kv-resident", "32768",
    "--vision",
    "--vram-reserve-mib", "700",
    "--trim-stage-weights",          // v0.1.39: 取代旧的 --stage-weights
    "--suffix-draft", "8",
    "--conversation-cache-mib", "24576",  // 会话停车（parking）：2026-10-11 由 8192 调大，见 §7
    "--conversation-cache-slots", "4",
    "--batch", "2", "--batch-groups", "2", // 两个 agent 真并行
    "--pipeline-windows", "2",             // #910：双卡 decode 流水线（--batch 下被引擎自动禁用）
    "--adapt-async", "1"                   // #910：resident RAM 模式（本配置下自动禁用）
  ],
  "layer_split": "24",
  "gpu": [0, 1],
  "port": 8000,
  "api_key": "llama_local",
  "env": {
    "STRATA_GR_V3": "1",
    "STRATA_BF16_TC": "1",           // ★ 必需，见 §5
    "STRATA_ATTN_MERGE_V2": "1",     // #910
    "STRATA_MTP_KV": "f16",          // #910
    "STRATA_PL_PLE_PREFETCH": "1",   // #910
    "STRATA_PL_PLE_LATE": "1",       // #910
    "STRATA_PL_EARLY_CHAIN": "1"     // #910
  }
}
```

## 4. 编译（从源码）

v0.1.39 的 `third_party` 只 vendored 了 `third_party/ggml`；**不要**传 `-DSTRATA_GGML_DIR`
（CMake 会按 `third_party/ggml/VERSION.txt` 自动拉取 pin 的 llama.cpp）。

```sh
cd ~/strata
rm -rf build
cmake -G Ninja -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
cmake --build build --target strata -j16
cp -f build/strata engine/strata
```

> 视觉编码器 `engine/strata-vision` 是独立 CMake 工程（`tools/vision`），沿用旧部署的即可；
> 需要时按 `MERGE-NOTES.md §6` 重编。

## 5. ★ 关键：sm_75 必须 `STRATA_BF16_TC=1`

v0.1.39 的 `src/prefill/gemm.cu` 在 **compute capability 7.5** 上默认**关闭** BF16→FP16 的
prefill 路径（`return cc < 75 ? 1 : 0`），本机因此 **prefill 掉 ~13–16%**。
配置里加 **`"STRATA_BF16_TC": "1"`** 即恢复。

## 6. 实测性能（本机）

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
- `--conversation-cache-mib 24576`（parking）：会话切走时快照到 RAM、回来时恢复，**后续轮次不重读历史**。
  实测（2026-10-11，本机 93 GiB RAM）：100K 会话首轮 100092 tok / 53.1s（1884 tok/s），
  追问 `100087 reused + 38 read` **1.7s**（整段重读要 ~52s）。
- **停车快照 ≈ 790 MiB + 15.0 KB/token**（实测：100K → 2.20 GiB，200K → 3.63 GiB，262K 约 4.4 GiB）。
- **池子必须装得下工作集，否则驱逐 → 整段重读**（实测，池 8 GiB）：
  4 个 100K 会话（4×2.36 = 9.4 GiB）逐个回访 → **#2/#4 `0 reused + 1001xx read`（各 54s）**，只有 #1/#3 复用。
  同一测试换池 24 GiB → 4 条全部 `1000xx reused + 38 read`（1.1–1.9s），**`evictions=0`**。
  → 2026-10-11 由 8192 调大到 **24576**。该参数是**预算不是预分配**（不用不占 RAM），
  `--conversation-cache-min-free-mib 2560` 兜底（RAM 不够时 skip parking，不会 OOM）。
- **已结案（原“待 96 GB 后复测”那两条）**：93 GiB 下 parking 稳定生效；2×200K 并发回访复用率 100%，
  不再出现 v0.1.39 的整段重读（2×100K 总 wall 207s）。
- **注意**：`--pipeline-windows` 与 `--adapt-async` 都**不能**和 `--batch` 共存（源码硬互斥，
  `not with --batch slots`；`--adapt-async` 还需驻留 RAM 模式，而该模式**不支持层切分**）——
  跟内存无关，加内存也不会解锁。

## 8. 与旧部署的差异 / 回退

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
- **回退引擎**：`engine-bak-opt/strata-v0138-batch559`。
- **GitHub**：`main` = 本版本；旧版在 `v0.1.38-legacy`。

## 9. 非 git 资源（换目录/重装时要单独带）

| 路径 | 说明 |
|---|---|
| `engine/strata`、`engine/strata-vision` | 本地编译的引擎（sm_75） |
| `build/`、`build-vision/` | CMake 构建目录 |
| `strata-swift-iq3_xxs.json` | 主配置（gitignore） |
| `mtp-rt/` | MTP 目录（**multi 草稿词表** + 软链到 `Strata-data/mtp/rt`）；en 备份 `draft_vocab.bin.en.bak` |
| `data/expert-profile.bin` | 专家 profile |
| `.bench/` | 基准脚本/请求/结果 |
| `engine-bak-opt/`、`engine-bak-*`、`old-local-backup/` | 回退用备份 |

## 10. 上游同步

- 本分支 = `origin/main`（v0.1.39）+ **#742/#743**（开放 PR）+ 本地文档。
- 上游合并 #742/#743 后：`git rebase origin/main` 时 `--skip` 掉它们。
- 上游出新版本时的完整同步步骤见 [`MERGE-NOTES.md`](MERGE-NOTES.md) 与 [`LOCAL-DELTAS.md`](LOCAL-DELTAS.md)。
