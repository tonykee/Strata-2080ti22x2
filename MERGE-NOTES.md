# 合并与同步说明（`--stage-weights` 移植到 v0.1.34）

> 本文件说明 `~/strata` 相对上游 `Niko1221/Strata` 做了什么改动、改动在哪、
> 以及**主版本更新时如何把这套填充优化重新同步进来**。
> 运行/实测数据/回退见同目录 [`PREFILL-PORT.md`](PREFILL-PORT.md)。

---

## 0. 一句话

只移植了上游缺少的一个能力：**`--stage-weights`**（双卡 layer split 时，每张卡只加载
自己那几层的 dense 权重，不再各存一份完整副本）。这是来自 `@spideytznn/Strata` 分支
`strata-2080tix2` 的改动。其余 fork 特性（`--stage-kv`、`--pin-resident-experts`、
resident-RAM 等）**上游 v0.1.34 已有更完整的实现**，没有移植（原因见 §3）。

---

## 1. 基线信息

| 项 | 值 |
|---|---|
| 上游 | `https://github.com/Niko1221/Strata` |
| 基线提交 | `origin/main` = **v0.1.38**（`99f3dbd`）；2026-10-03 从 v0.1.34 同步上来 |
| 本仓库分支 | `port-v0134`（名字保留；内容是 v0.1.38 + 补丁） |
| 本地提交 | **`a15d062`** `LOCAL: --stage-weights ...`；**`c006996`** `LOCAL setup: pin gcc-13 ...` |
| cherry-pick 的上游 PR | **`66f4945`** = #593（BF16→FP16 tensor core, sm_75）；**`977a9fa`** = #575（qsa top-k 拆分）。见 §9 |
| 补丁文件 | `local-patches/0001-LOCAL-stage-weights-...patch`；`local-patches/0001-LOCAL-setup-pin-gcc-13-...patch` |
| 构建 | CUDA 12.8 / gcc-13 / sm_75，见 §6 |

> **2026-10-03 从 v0.1.34 rebase 到 v0.1.38**：唯一冲突在 `src/program/generate.cpp`
> （上游 #486 的显存不足报错 × 我们的 `stage_pool_bytes`），已合并为"用 stage 尺寸 + 保留 #486 报错"。
> `weights.hpp`/`weights.cpp`/`native_dense.cpp` 上游未动，补丁干净套用；`setup.py` 的 gcc-13 补丁自动合入。

> 上游 remote 就是 `~/strata` 的 `origin`（Niko1221/Strata）；`git fetch` 后
> `git rebase origin/main` 即可重放本地补丁。

---

## 2. 改了什么（提交 `a15d062`，7 个文件，+82/−13）

### 2.1 新增：`include/strata/core/weight_stage.hpp`
一个按层区间判断张量归属的小结构体 `WeightStage{begin,end}`：
`blk.N.*` 属于 `[begin,end)`；非 `blk.` 前缀的全局张量（embedding、output、PLE key 等）
一律保留。名字非法时保守返回 true（不丢张量）。

### 2.2 `include/strata/core/weights.hpp` / `src/core/weights.cpp`
- `WeightRef` 增加字段 `bool stage_resident = true;`（仅被本 stage 排除的层为 false）。
- `WeightTable::pool_bytes(..., const WeightStage* stage = nullptr)`：
  计算 arena 大小时，把 `!stage->owns(name)` 的行也计入“紧凑排除”。
- `WeightTable::load(..., const WeightStage* stage = nullptr)`：
  加载时被排除的行 `skipped[i]=true`，保留元数据、不占 arena 字节，
  并置 `wr.stage_resident = false`。默认参数保证老调用点不变。

### 2.3 `src/core/native_dense.cpp`
`NativeDense::load` 里跳过 `!ref.stage_resident` 的张量：其它卡的 native 投影不上传。

### 2.4 `src/core/verify.cpp` / `src/prefill/prefill.cpp`
`need()` 遇到 `stage_resident == false` 时明确报错
`"<name> belongs to another GPU stage"`，而不是含糊的 `is missing`。

### 2.5 `src/program/generate.cpp`（唯一较大的改动）
- 选项 `bool stage_weights`，解析 `--stage-weights`，help 文本。
- 校验：必须 `native --serve`、`--spec >= 2`、**显式多卡 `--layer-split K`（不接受 `auto`）**，
  且不能与整模型图诊断（`--gpu-only-full`/`--graph-only`/`--gpu-stages`）同用。
  原因：later stage 的 arena 在 auto 切分搜索**之前**就要定尺寸，所以必须显式 K。
- 主卡加载：`main_weight_stage{0, split_at.front()}` → `pool_bytes/load`。
- 各 later stage：`stage_range{split_at[i], (last?n_layers:split_at[i+1])}`，
  `stage_pool_bytes` 单独计算并按此 `cudaMalloc` + `load`。
- 日志：`--stage-weights: CUDA0 loads layers 0-23 only (771.01 MiB)` 等，便于确认生效。

> 注：验证用的 `need()` 保护是必需的——因为 `wt` 现在**故意**不含其它层的张量，
> 任何“跨卡误读”必须变成清晰错误而不是空指针。

---

## 3. 为什么只移植这一个（不要重复移植的东西）

对照上游 v0.1.34，fork 的其它优化**已经存在或已被更好的实现取代**：

| fork 特性 | 上游 v0.1.34 状态 | 结论 |
|---|---|---|
| `--stage-kv`（本卡只分配本层 QSA 状态） | 0.1.30 起有 layer-range session carve（`77709ef`），**QSA+GDN 都切**，接入快照/checkpoint | 不要移植 |
| `--resident-cpu-experts` + 动态交换 | 已有，且更完整（`--resident-pin`/`--resident-headroom`/`--resident-budget-gib`/partial pin/`exchange_cache_complement`） | 不要移植 |
| `--pin-resident-experts`（锁页直传） | 已有：`--resident-pin` + `g_pinned_share` 直传环；且本机 40 GiB arena 在 Linux 上全量 `cudaHostRegister PORTABLE`，prefill 本就走直传 | 不要移植 |
| 文件层预取 | 0.1.31 起有 routing-aware 预取 | 不要移植 |
| **`--stage-weights`** | **没有**；上游注释明确「每个 stage 保留完整 dense 权重副本」 | **移植** |

因此主版本再有更新时，**只需重放这两个本地提交**（`a15d062` + `c006996`）。

---

## 4. 主版本更新后的同步步骤（重点）

### 方式 A：rebase（推荐，改动是一个提交）

```sh
cd ~/strata
git fetch origin
git checkout port-v0134
git rebase origin/main          # 把 a15d062 / c006996 重放到新上游
# 解决冲突（见下）后：
cmake --build build --target strata -j16
cp -f build/strata engine/strata
```

### 方式 B：补丁重放（当分支历史乱了）

```sh
cd ~/strata
git fetch origin
git checkout -B port-v0134 origin/main
git am local-patches/0001-LOCAL-stage-weights-for-a-multi-GPU-layer-split.patch
cmake --build build --target strata -j16 && cp -f build/strata engine/strata
```

### 冲突热点（按概率排序）

1. **`src/program/generate.cpp`（最可能冲突）**——上游改得最频繁（0.1.31→0.1.34 就 +1405/−393）。
   需要重定位的 4 处：
   - 选项字段 `stage_weights` 与解析行 `--stage-weights`；
   - `multi_gpu` 之后的校验块；
   - 主卡 `pool_bytes/load` 两个调用点（在 `std::set skip` 之后）；
   - later stage 循环里的 `stage_range` / `stage_pool_bytes` / `cudaMalloc` / `load`。
2. **`src/core/weights.cpp` / `weights.hpp`（通常很稳）**——`pool_bytes`/`load` 是纯 loader，
   历史上很少动；若冲突，按 §2.2 的语义手工加 `stage` 分支即可。
3. `native_dense.cpp`、`verify.cpp`、`prefill.cpp` 各只有 1 行，冲突易解。

### 同步后必须验证（两条日志）

```sh
cd ~/strata && ./start_iq3.sh        # 端口 8000
grep -E "stage-weights: CUDA[01] loads layers" strata-swift-iq3_xxs.log
# 期望：
#   strata generate: --stage-weights: CUDA0 loads layers 0-23 only (771.01 MiB)
#   strata generate: --stage-weights: CUDA1 loads layers 24-47 only (708.31 MiB)
```

另外 `grep -n "stage-weights" <(./engine/strata --help)` 应能看到帮助行；
`--layer-split auto` 应被拒绝。

### 若上游未来自己实现了 per-stage dense 权重

那时可直接删掉本补丁，改用上游的开关；先在 `--help` 里搜 `stage`，
并检查上游是否仍让每个 stage 各存一份 dense 权重（搜 `each later stage's own copy`）。

---

## 5. 关键语义 / 约束

- `--stage-weights` 只影响 **dense（稠密）权重**；专家、量化格式、ngram、KV 都不变，
  数值等价（只是不再重复加载别的层）。
- 全局张量（`token_embd.weight`、`output.weight`、`blk.1.ple_key.weight` 之外的非 blk
  与各 stage 自己层）按 `WeightStage::owns` 决定；PLE key 在第 1 层，落在 stage 0，
  因此要求 `K >= 2`（`--layer-split` 本身要求 rising K from 2）。
- 必须显式 `--layer-split K`：auto 切分在权重加载后才搜索，无法提前定 stage 尺寸。
- 与 `--kv-resident`、`--vision`、`--expert-cache auto` 均兼容（本机实测组合）。

---

## 6. 编译

```sh
cd ~/strata
cmake -G Ninja -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DSTRATA_GGML_DIR=/home/likan/strata/third_party/llama.cpp -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
cmake --build build --target strata -j16
cp -f build/strata engine/strata

# 视觉编码器（独立 CMake 工程）
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=/home/likan/strata/third_party/llama.cpp -DSTRATA_VISION_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
cmake --build build-vision --target strata-vision -j16
cp -f build-vision/bin/strata-vision engine/strata-vision
```

---

## 7. 实测（摘要，详见 `PREFILL-PORT.md`）

Swift IQ3_XXS，256K，kv-resident，视觉开，2× RTX 2080 Ti 22G：

| 配置 | 12.5K | 49.7K |
|---|---:|---:|
| **v0.1.34 + `--stage-weights` + `--prefill auto`（推荐）** | 934.8 | **1313.3** |
| 同上，无 `--stage-weights` | 919.9 | 1279.2 |
| 同上 + `--no-prefill-borrow` | 970.5 | 1053.2 |
| 旧 v0.1.31 `--prefill 1024`（参考） | ~425–576 | — |

`--stage-weights` 效果：CUDA1 空闲显存 13.7→19.4 GiB，驻留专家 19036→21141，长 prompt +2.7%。

---

## 8. 回退

`--stage-weights` 默认关闭；不加就是原版 v0.1.34。要回到旧 v0.1.31 部署：
`cd ~/strata && git checkout main`（代码回到 v0.1.31 + gcc-13 补丁），旧引擎二进制在
`engine-bak-0131/`；或 `git checkout port-v0134` 切回本部署。分支 `main` 与 `port-v0134`
都保留，随时可切。

---

## 9. 合入的上游 PR（#593 / #575）

两个上游**开放** PR 以 cherry-pick 合入（都基于 v0.1.38，单提交，改动小）：

| 提交 | PR | 作用 | 实测（本机 IQ3_XXS） |
|---|---|---|---|
| `66f4945` | [#593](https://github.com/Niko1221/Strata/pull/593) | BF16 投影（router/indexer/GDN/hyper-connection）走 **FP16 tensor core**（sm_70/sm_75）；复用 prompt 缓冲 + dequant scratch | prefill 12.5K **+19%** |
| `977a9fa` | [#575](https://github.com/Niko1221/Strata/pull/575) | qsa top-k 超寄存器容量（~135K cells）时拆分 + 合并 | 长 prompt 叠加到 **+22%** |

**内存中性**：合入前后启动驻留/缓存/剩余显存逐项一致（21141 专家、9283 槽、472 MiB、21529/21510 MiB）。

### 重新获取 / 移除

```sh
cd ~/strata
git fetch origin pull/593/head:pr593 pull/575/head:pr575   # 重新取
git rebase -i origin/main                                  # 交互式，删掉这两个提交即可移除
```

**上游若合并了它们**：下次 `git rebase origin/main` 时这两个提交多半会变成空提交或被判为已应用——直接 `git rebase --skip` 跳过即可（我们自己的两个 `LOCAL` 补丁继续保留）。

### 试过但未采用（本机无收益）

| PR | 提交 | 作用 | 实测 |
|---|---|---|---|
| [#500](https://github.com/Niko1221/Strata/pull/500) | `9627bf8` `d728f36`（已撤） | CPU 专家池中间激活量化并行化（消 phase5→6 屏障） | prefill 12.5K +1.0%、49.7K +0.2%、193K −0.3%；decode 不变 —— **≈0**，已回退 |
| [#547](https://github.com/Niko1221/Strata/pull/547) | `0de400d`（已撤） | prefill `bytes_needed` 按 carve 实算（prompt 借用 3.94→3.78 GiB） | 填充/decode 无变化，已回退 |
| [#583](https://github.com/Niko1221/Strata/pull/583) | `54dde7b` `a2feb06` `9878134`（已撤） | ring 改按字节预算（分卡 ring 96→227 槽）、`--prefill auto` 改二分选块 | 裸 `auto`：12.5K +1.4% / 49.7K +1.0% / 193K −0.4%（噪声内）；`auto:16384`：49.7K **−11%**、193K **−5%**（更大块更差）。**无明显收益，已回退** |

> 两者都在噪声内；按「只保留有实测收益的改动」原则移除（`git reset --hard 864bf68` 后仅重放 #589）。

### 保留但需配套才生效

| PR | 提交 | 作用 |
|---|---|---|
| [#589](https://github.com/Niko1221/Strata/pull/589) | `9e3940d` | `make_profile --reorder`：使用轨迹可排在完整基础排名之前 → **需重新生成 `expert-profile.bin` 才有运行收益**（当前未重新生成，故未测） |


