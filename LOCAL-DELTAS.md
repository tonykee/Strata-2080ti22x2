# 本地 vs 上游差异清单（便于同步主干）

> 本文件列出 `~/strata`（分支 `port-v0134`）相对上游 `Niko1221/Strata` 的**全部差异**，
> 供将来 `git rebase origin/main` 时逐项处理。
> 深度说明见 [`MERGE-NOTES.md`](MERGE-NOTES.md)；实测见 [`PREFILL-PORT.md`](PREFILL-PORT.md)。

## 0. 基线

| 项 | 值 |
|---|---|
| 上游 remote | `origin` = https://github.com/Niko1221/Strata |
| 上游基线 | `origin/main` = **`fb58e0db`**（v0.1.41 线，2026-10-08） |
| 部署分支 | **`v0.1.39-2080ti`**（= `main`，自 2026-10-06；= v0.1.39 + #742/#743 + #910 + multi 词表） |
| 备用分支 | `backup-main-20261010`（提升 #1659 前的 main）、`port-v0134`、`batch-559-646` |
| 领先上游 | 本地领先 **862** 个提交；上游领先本地 **1552**（两条线已深度分叉，只能挑摘） |

> **2026-10-04 优化评估**：见 [`OPT-2026-10.md`](OPT-2026-10.md)。新增 B 类补丁 **#650**（启动 −10 s）
> 与 **#646**（decode **+13%**）；#655/#699/#663/#704/#693/#651 已跳过或回退。
>
> **2026-10-04 #559 合并**：候选分支 `batch-559-646` 把上游开放 PR **#559**（batch slots，2..8 会话并发）
> 合到 #646 上并验证（batch 输出与 solo 逐 token 相同；双并发各 ~27 tok/s、聚合 58.9 rows/s；见
> [`OPT-2026-10.md`](OPT-2026-10.md) §6）。**注意**：该分支用 #559 的 `--trim-stage-weights` **取代**了
> 本地 A 类的 `--stage-weights`（见 §1 说明）。

> **2026-10-08 / 10-10 更新**：
> - 部署分支 = **`v0.1.39-2080ti`**（= `main`），上游基线 = `origin/main` **`fb58e0db`**（v0.1.41 线）。
> - 2026-10-08 评估的上游 10-07 后 PR **全部不采纳**（#1441/#1367/#1251/#1099、peer tier），见
>   [`OPT-2026-10.md`](OPT-2026-10.md) §9 与 §3 的 C 类清单。
> - 2026-10-10 **收下 #1659**（B 类，FP16 GEMM 的 cuBLAS status-14 保险，见 §2）——这是本 fork 唯一采纳的
>   新增上游提交；同日确认 `#1237`（pinned stage buffers 会弄坏 IQ3_S decode）与 batch-MTP draft head 修复
>   **都不适用于本 fork**。

> **注意**：rebase 会重写提交哈希，下面的哈希只是当前值；同步时按**提交主题**识别。

---

## 1. 必须保留、每次同步都要重放的本地补丁（A 类）

| 提交 | 主题 | 改动文件 | 说明 |
|---|---|---|---|
| `a15d062` | `LOCAL: --stage-weights for a multi-GPU layer split` | `include/strata/core/weight_stage.hpp`（新）、`include/strata/core/weights.hpp`、`src/core/weights.cpp`、`src/core/native_dense.cpp`、`src/core/verify.cpp`、`src/prefill/prefill.cpp`、`src/program/generate.cpp` | 双卡 layer split 时每卡只加载自己层的 dense 权重（腾显存给专家缓存）。上游 v0.1.38 仍无此功能 |
| `c006996` | `LOCAL setup: pin gcc-13 as the CUDA host compiler` | `setup.py` | CUDA 12.8 拒绝 gcc-15；固定 gcc-13 + `--allow-unsupported-compiler`。**当前默认 gcc-14 也能编，但防默认变 15** |

补丁文件：`local-patches/0001-LOCAL-stage-weights-*.patch`、`local-patches/0001-LOCAL-setup-pin-gcc-13-*.patch`。

> **候选分支 `batch-559-646` 的变化**：合入 #559 后，`a15d062` 的 `--stage-weights` 被 **#559 的
> `--trim-stage-weights` 取代**（两者功能等价；#559 通过 `NativeDense::load(..., layer_lo, layer_hi)` 实现）。
> 该分支**移除**了 `--stage-weights` 的选项/校验/加载代码，配置需改用 `--trim-stage-weights`。`port-v0134`
> 仍保留 `--stage-weights`。

---

## 2. cherry-pick 的上游开放 PR（B 类；上游合并后应 `--skip` 掉）

| 提交 | PR | 改动文件 | 作用 / 实测 |
|---|---|---|---|
| `66f4945` | [#593](https://github.com/Niko1221/Strata/pull/593) | `include/strata/prefill/gemm.hpp`、`src/prefill/gemm.cu`、`src/prefill/prefill.cpp` | BF16 投影走 FP16 tensor core（sm_75）；prefill +19% |
| `977a9fa` | [#575](https://github.com/Niko1221/Strata/pull/575) | `include/strata/kernels/qsa_select.hpp`、`src/kernels/cuda/qsa_select.cu`、`src/kernels/qsa_topk_active_parity.cpp` | qsa top-k 超寄存器容量时拆分；长 prompt 叠加到 +22% |
| `9e3940d` | [#589](https://github.com/Niko1221/Strata/pull/589) | `tools/make_profile.py`、`tools/test_make_profile.py` | `make_profile --reorder`；**需重建 profile 才有运行收益** |
| `f8ad1fb` | [#650](https://github.com/Niko1221/Strata/pull/650) | `src/core/pinned.cu` | Linux arena 用透明大页；本机启动 **116→105 s**，decode 不变 |
| `b533132` `08fbf04` | [#646](https://github.com/Niko1221/Strata/pull/646) | 22 文件（`verify.cpp`、`mtp.cpp`、`iq_kernels.cu`、`shared_expert.cu` 等） | decode **+13%**（sm_75）、贪心输出逐字节相同；**`qsa_select.cu` 取 #575 版本**（两套超容量方案互斥） |
| `4491149`（合并提交） | [#559](https://github.com/Niko1221/Strata/pull/559) | 22 文件（`verify.cpp`、`generate.cpp`、`server.py`、`conversation_*`、`tools/autoconfig.py` 等） | **多用户并发**（`--batch 2..8`）；batch 输出与 solo 逐 token 相同；本机双并发聚合 58.9 rows/s。**取代本地 `--stage-weights`**（见 §1）；仅在候选分支 `batch-559-646` |
| `c79fba42` | [#1659](https://github.com/Niko1221/Strata/pull/1659) | `include/strata/prefill/gemm.hpp`、`src/prefill/gemm.cu` | **保险**：进程里有第二个 CUDA context 时，cuBLAS 12 的 FP16 **默认算法**对窄形状返回 `CUBLAS_STATUS_INTERNAL_ERROR`(14) → 记住失败形状、只对这些改走 `CUBLAS_GEMM_ALGO2`（issue #1650，Turing + Pascal 实测）。**本机默认 `--short-read 64` 天然免疫**（最后 ≤64 token 走 decode windows，不产生 ~6 token 的角色头小 chunk）；实测性能无变化。**上游合并后 `--skip`** |
| `0766290a` | [#1821](https://github.com/Niko1221/Strata/pull/1821) | `src/kernels/cuda/qsa_prompt_attn.cu`、`src/kernels/qsa_prompt_attn_parity.cpp` | **Turing + `--kv int8` 专属**：prompt attention（v1）的 int8 K/V 预取进寄存器 + K/V 共享同一块 SMEM（37.9→29.2 KB → 每 SM 1→2 块）。本机交替 A/B：prefill 15K **+3.9%**、130K **+5.1%**，decode 持平；**bit-identical**。**上游合并后 `--skip`** |

重新获取：
```sh
git fetch origin pull/593/head:pr593 pull/575/head:pr575 pull/589/head:pr589 \
                 pull/650/head:pr650 pull/646/head:pr646 pull/559/head:pr559 \
                 pull/1659/head:pr1659 pull/1821/head:pr1821
```

---

## 3. 试过但已回退（C 类；无需保留，仅记录，避免重复评估）

| PR | 结果 |
|---|---|
| [#500](https://github.com/Niko1221/Strata/pull/500) CPU 池量化并行 | ≈0（噪声内），已回退 |
| [#547](https://github.com/Niko1221/Strata/pull/547) bytes_needed 少借显存 | 无变化，已回退 |
| [#583](https://github.com/Niko1221/Strata/pull/583) ring 字节预算 + auto 二分 | 裸 auto 噪声内；`auto:16384` −11%/−5%，已回退。**`--prefill` 保持 `auto`，勿调大** |
| [#655](https://github.com/Niko1221/Strata/pull/655) sm_75 BF16→FP16 tensor core | 与本地 #593 同文件同功能，冗余，未叠加（#593 已 +19%） |
| [#699](https://github.com/Niko1221/Strata/pull/699) Linux 启动预读 | 本机 SATA SSD 顺序读已到顶（0.51 GiB/s），启动无变化，已回退 |
| [#663](https://github.com/Niko1221/Strata/pull/663) layer-split 空闲卡帮短 prompt | 本机 1.34K +1.9%、2.59K **−11.7%**（86% 专家常驻，跨卡搬运净亏），已回退 |
| [#693](https://github.com/Niko1221/Strata/pull/693) prefill 分块细化 + 等长块 | auto 持平、`auto:16384` **−9%**（49.7K 尾块<8192 触发不了等长块），已回退 |
| [#704](https://github.com/Niko1221/Strata/pull/704) HC_Q8 超连接 int8 | 明确 “Not with a layer split”，本部署不适用 |
| [#651](https://github.com/Niko1221/Strata/pull/651) PLE 表 Q5_1/Q8_0 | 兼容性补丁，本机 PLE 表格式已支持，无提速 |
| [#1441](https://github.com/Niko1221/Strata/pull/1441) `STRATA_PREFILL_PIPE` 层切分 chunk 尺寸 | **2 段切分下 15K prefill −15%**（chunk 8192→3840，4 段）；≥64K 等于没变（算出的 chunk 就等于 cap）。作者在 4 段切分上测的，已回退（2026-10-08） |
| [#1367](https://github.com/Niko1221/Strata/pull/1367) `STRATA_PROMPT_ATTN_IMMA` int8 张量核 prompt attention | 速度中性（1238/1223 vs 1250/1233），仅数值更接近 FP32；未采纳（2026-10-08） |
| [#1251](https://github.com/Niko1221/Strata/pull/1251) 无 P2P 的 peer prompt share | `--peer-device` 与 `--layer-split` **互斥**；本机有 NVLink（NV2），不需要 host route。不适用（2026-10-08） |
| [#1099](https://github.com/Niko1221/Strata/pull/1099) 层切分 hand-off 走 NVLink P2P | 依赖上游 **#796 part A**（`bytes_needed_impl`/`bytes_needed_owned` 重构，本 fork 无）→ `prefill.cpp` 结构性冲突；收益仅 +0.9%，未采纳（2026-10-08） |
| `--peer-device 1`（peer tier） | prefill **−26%（15K）/ −50%（64K）**、decode +3%：GPU0 要扛全部 48 层 → 缓存 8651→5737 槽，且 peer 的 prompt 缓冲没放下（GPU1 闲置）。两卡对等时不划算，已回退（2026-10-08） |
| [#1720](https://github.com/Niko1221/Strata/pull/1720) Q8_0 dense 反量化合并访问 | **无收益**（效应在 A/B 漂移以内 ≈0）。PR 的"7×"是 **V100 上核的写带宽**，不是端到端；2080 Ti 上 Q8_0 dense 反量化本来就不是瓶颈（2026-10-10） |
| [#1660](https://github.com/Niko1221/Strata/pull/1660) 驻留层 MMQ 按行数排序（+ 依赖 `c6a0d403`） | **前提不成立**：要求该层 512 个专家**全部常驻**，本机每层有 **144–327 个不在显存**（"99.6% 路由质量"≠"全专家常驻"）→ 排序从不执行。已回退（2026-10-10） |

---

## 4. 本地文档 / 脚本 / 补丁（D 类；与上游无冲突，可保留或重建）

| 文件 | 说明 |
|---|---|
| `LOCAL-DELTAS.md` | 本文档：本地 vs 上游差异总清单 |
| `MERGE-NOTES.md` | 合并原理 + 同步步骤 + 冲突热点 |
| `PREFILL-PORT.md` | 填充/解码实测、调优、运行/重编/回退 |
| `DFLASH2-STATUS.md` | DFlash2 上游状态 + 训练可行性 |
| `local-patches/*.patch` | A 类补丁的独立文件 |
| `start_iq3.sh` | 主部署启动脚本（端口 8000、主配置） |

---

## 5. 非 git 的部署资源（不在上游、也不在 git 里）

> 这些是**运行态**，`git clone` 不会带；换目录/重装时需单独复制或重建。

| 路径 | 说明 |
|---|---|
| `engine/strata`、`engine/strata-vision` | 本地编译的引擎（sm_75） |
| `build/`、`build-vision/` | CMake 构建目录 |
| `strata-swift-iq3_xxs.json` | 主配置（被 `.gitignore` 忽略；端口 8000、`--stage-weights`、`--layer-split 24`、`--prefill auto`、`--kv-resident 32768` 等） |
| `mtp-rt/` | 隔离的 MTP 目录（en 草稿词表 + 软链到 `Strata-data/mtp/rt` 的权重） |
| `.bench/` | 基准脚本/请求/结果；`.bench/configs/` 存实验配置 |
| `engine-bak-*/`、`old-local-backup/` | 回退用的旧引擎/旧文件 |
| `IQ3_S-NOTES.md`（未跟踪） | 你/另一会话留下的 IQ3_S 部署留底，未提交；不被本分支改动 |
| `backup-*` 分支 | `backup-before-583`、`backup-with-593-575`、`backup-port-v0138-pre-prs`、`backup-port-v0134`、`backup-main-0131`（= `main`） |

---

## 6. 同步主干的操作（可复制）

```sh
cd ~/strata
git fetch origin
git branch -f backup-pre-sync HEAD          # 安全备份
git rebase origin/main                       # 重放 A/B 类提交
# 冲突热点：
#   - src/program/generate.cpp（A 类 stage-weights 的几处插入；B 类 #583 曾覆盖此处）
#   - setup.py（A 类 gcc-13）
#   - src/prefill/prefill.cpp（A 类 1 行 need 保护 + B 类 #593）
# 若某个 B 类提交被上游已合并 → git rebase --skip 跳过它
cmake --build build --target strata -j16
cmake --build build-vision --target strata-vision -j16
cp -f build/strata engine/strata
cp -f build-vision/bin/strata-vision engine/strata-vision
# 重启服务（8000）
```

### 同步后验证清单

- [ ] `git grep -n WeightStage src include` → stage-weights 在
- [ ] `grep -n gcc-13 setup.py` → gcc-13 补丁在
- [ ] 引擎日志出现 `--stage-weights: CUDA0 loads layers ... only` 与 `CUDA1 loads layers ... only`
- [ ] `curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer llama_local' http://127.0.0.1:8000/v1/models` → 200
- [ ] 抽查填充：12.5K 应 ≈1100 tok/s、49.7K ≈1560（若明显变慢，回退）
