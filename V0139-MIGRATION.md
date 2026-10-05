# 迁移到 v0.1.39（分支 `v0.1.39-2080ti`）

> 2026-10-05。把主部署从 **v0.1.38 + cherry-pick** 迁到 **上游 v0.1.39 + #742/#743**。
> 机器：2× RTX 2080 Ti 22G（sm_75）、61 GiB RAM、SATA SSD、CUDA 12.8 / gcc-13。

## 1. 为什么迁

上游 **v0.1.39**（`6f32ec0`）把我们之前 cherry-pick 的 PR **全部并入**（重做版）：

- #646 decode、#559/#465 多并发 + parking + layer split、#575（被 #646 的 `capacity_guard` 取代）、
  #650 透明大页、#593 BF16→FP16、#589 make_profile、#653/#716 parking。

所以迁移后**不必再维护那些 cherry-pick**，只保留开放 PR **#742/#743**（见下）。

## 2. 一个关键坑：sm_75 上 BF16→FP16 被默认关掉

v0.1.39 的 `src/prefill/gemm.cu`：

```cpp
if (cc <= 0 || cc >= 80) return 0;
if (cc < 70) return 2;
return cc < 75 ? 1 : 0;   // cc=75（2080 Ti）→ 0 = 关闭
```

→ **本机 prefill 掉 ~13-16%**。修复：配置里加 **`STRATA_BF16_TC=1`**（强制开启）。

## 3. 叠加 #742 + #743（Turing 长 prompt prefill）

两个开放 PR，都是 qsa_select 的长 prompt 优化：

- **#742** `qsa_select: FP32 tiled block scores below sm_80`
- **#743** `qsa_select: coalesced streaming top-k for long Turing prompts`

本机实测（`--batch 2`，fresh，`STRATA_BF16_TC=1`）：

| prefill | v0.1.38+补丁（旧部署） | v0.1.39 默认 | **v0.1.39 + #742** | **+#742+#743** |
|---|---:|---:|---:|---:|
| 12.5K | 1047.7 | 929.3 | 1119.9 | ~1110 |
| 49.7K | 1497.5 | 1299.3 | 1655.8 | ~1640 |
| **192.8K** | 1201.9 | 1223.1 | 1739.3 | **1785.4** |

部署实测 192.8K：**1860.5 tok/s**（旧部署 1201.9，**+55%**），读入 158s → **105s**。

decode：引擎速度与旧部署**持平**（隔离测试 `--spec 2`：~63.6 vs ~62.6）；完整配置下看到的差异来自
**生成内容不同**（prompt 路径舍入）导致草稿命中不同，不是引擎快慢。

## 4. 配置

`strata-swift-iq3_xxs.json`（gitignore 文件）：

```diff
-  "--stage-weights",
+  "--trim-stage-weights",
+  "--conversation-cache-mib", "8192",
+  "--conversation-cache-slots", "4",
+  "--batch", "2",
+  "--batch-groups", "2",
```
env 增加 `"STRATA_BF16_TC": "1"`（保留 `STRATA_GR_V3: "1"`）。

## 5. 编译（v0.1.39 的 third_party 布局变了）

v0.1.39 vendored 的是 `third_party/ggml`，**不要**传 `-DSTRATA_GGML_DIR`（它会按 `VERSION.txt` 拉取 pin 的 llama.cpp）：

```sh
rm -rf build
cmake -G Ninja -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
cmake --build build --target strata -j16 && cp -f build/strata engine/strata
```

## 6. 已知 / 待办

- **并发“重读”**：v0.1.39 的“slot 只剩一个请求就回 solo 路径”行为，在 61 GB（parking 被跳过）时会让该请求
  **整段重读**（实测 2×100K 总 wall 207s vs 旧部署 171s）。**96 GB 下 parking 生效应能避免**——升内存后再验证。
- **#575** 未进 v0.1.39（被 #646 的 `capacity_guard` 取代）；本机长 prompt prefill 由 #742/#743 补上且更强。
- **#742/#743 仍是开放 PR**：上游合并后 `git rebase --skip` 掉。

## 7. 回退

- 旧部署分支：`batch-559-646`（v0.1.38 + #646/#559/#593/#575/#650/#589）或 `port-v0134`。
- 旧引擎备份：`engine-bak-opt/strata-v0138-batch559`。
