# ~/strata-v0134 — v0.1.34 + `--stage-weights` 独立测试部署

> 本目录是**独立**部署，不影响 `~/strata`。基于上游 `origin/main` = **v0.1.34**
> （分支 `port-v0134`，提交 `c8e9bea` 起），本机为 2× RTX 2080 Ti 22G（sm_75）、
> 61 GiB RAM、CUDA 12.8、gcc-13。

## 1. 这轮做了什么

1. 按 `@spideytznn/Strata` 的 `strata-2080tix2` 分支，把**唯一在上游缺失的填充能力
   `--stage-weights`** 移植到 v0.1.34（其余如 `--stage-kv`、锁页直传 resident 等上游
   已有更完整实现，未移植）。改动见提交 `c8e9bea`。
2. 在 v0.1.34 上实测填充（Swift IQ3_XXS，256K，kv-resident，视觉开，`--layer-split 24`）。

## 2. 实测填充（prompt 读入速度，tok/s）

| 配置 | 12.5K | 49.7K |
|---|---:|---:|
| **v0.1.34 + `--stage-weights` + `--prefill auto`（默认）** | **934.8** | **1313.3** |
| 同上，无 `--stage-weights` | 919.9 | 1279.2 |
| 同上 + `STRATA_SPLIT_RING=384` | — | 1307.3 |
| 同上 + `--no-prefill-borrow`（独立缓冲） | 970.5 | 1053.2 |
| 参考：旧 v0.1.31 `--prefill 1024`（DEPLOY 记录） | ~425–576 | — |

- **长 prompt（≳25K）稳定 >1200**；49.7K 达 1313，已超过 fork 双 2080Ti 文档的 1192。
- `--stage-weights` 让每卡只加载自己层的 dense 权重：CUDA1 空闲显存 13.7→19.4 GiB，
  驻留专家 19036→21141，长 prompt +2.7%。
- 填充随 prompt 变长而升高（固定开销被摊薄、更多专家进入流水）。

## 3. 视觉验证（本部署）

红底 + 白色大写文字 `STRATA 12345`（`.bench/testimg.png`），经双卡 layer split：

- 问「背景什么颜色、文字是什么」→ 回答
  `The background of the image is red.` / `The text in the image reads exactly: STRATA 12345`
- `finish_reason: stop`，解码 ~62.5 tok/s，图片路径无回归。

## 4. 启动 / 停止

```sh
cd ~/strata-v0134
./start_iq3.sh                 # 端口 8000，配置 strata-swift-iq3_xxs.json
# 停止
./stop.sh 2>/dev/null || pkill -f "[s]trata-v0134/serve/server.py"; pkill -f "[s]trata-v0134/engine/strata"
```

服务接口（与旧部署相同，另需 API key `llama_local`）：

```sh
curl -s -H "Authorization: Bearer llama_local" http://127.0.0.1:8000/v1/models
```

## 5. 关键配置（`strata-swift-iq3_xxs.json`）

与旧 `~/strata` 的 Swift IQ3_XXS 配置一致，差别只有：

- `exe` → `/home/likan/strata-v0134/engine/strata`，`port` → `8000`
- `--prefill` 由 `1024` 改为 **`auto`**（自动选块，本机 8192）
- `layer_split` 由 `auto` 改为显式 **`24`**（`--stage-weights` 要求显式切分）
- 新增 **`--stage-weights`**

**不要**加 `--no-prefill-borrow`（长 prompt −20%）；**不要**加 `STRATA_SPLIT_RING`（无益）。

## 6. 重新编译

```sh
cd ~/strata-v0134
cmake -G Ninja -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DSTRATA_GGML_DIR=/home/likan/strata/third_party/llama.cpp -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
cmake --build build --target strata -j16
cp -f build/strata engine/strata
# 视觉：cmake -S tools/vision -B build-vision ... ; cp build-vision/bin/strata-vision engine/
```

## 7. 回退

`~/strata` 完全没动，直接 `cd ~/strata && ./start_iq3.sh` 即可回到原 v0.1.31 部署。

## 8. 代码/文档场景解码调优（无额外功耗/噪音）

用户主场景：读代码、写文档、写代码、写游戏。实测（IQ3_XXS，160W，双卡）：

| 调项 | 效果 | 备注 |
|---|---|---|
| `spec_min_p` 0.5 → **0.65** | +2.5%（接受率 73%→82%） | 越高越平；可请求级 `strata_tune` 覆盖 |
| draft 词表 CJK → **en**（`data/draft_vocab_en.bin`） | 代码 +1~2%，省 ~110 MiB 显存 | 只需换 `mtp-rt/draft_vocab.bin` |
| `--suffix-draft 8` | 写代码 +7.7%，改代码 +3.3% | `16` 反而变差 |
| `STRATA_GR_V3=1` | +1% | Turing 两半切分修复后可用 |

**固化的生产配置** `strata-swift-iq3_xxs.json` 已包含：隔离 MTP 目录 `mtp-rt/`（en 词表）、`--spec-min-p 0.65`、`--suffix-draft 8`、`env.STRATA_GR_V3=1`、`--stage-weights`、`--prefill auto`、`--layer-split 24`、端口 8000。

解码参考：中文对话 50–64 tok/s、读代码改代码 ~50、从零写代码 ~44–48；prompt 读入 210 tok/s（392-token）到 1313 tok/s（49.7K）。

**注意**：~50 tok/s 是这台卡在 160W 下的带宽/流水线瓶颈（GPU 仅 110–125W、两卡各约 50% 利用率、命中 99%+），软件调优空间只有个位数~十来个百分比，不要期待 2x。

