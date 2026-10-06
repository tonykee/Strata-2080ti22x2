# MTP 草稿词表：en（40525）→ multi（106299）（2026-10-06）

> 本机：2× RTX 2080 Ti 22G（sm_75）、61 GiB RAM、SATA SSD、CUDA 12.8 / gcc-13。
> 引擎：`v0.1.39-2080ti`（v0.1.39 + #742/#743 + #910）。分支 `v0.1.39-2080ti`。

## 0. 一句话

主部署（Swift）和 base IQ3_S 共用的 `mtp-rt/` 挂的是 **英文草稿词表（40525 个 token）**，中文输出时
草稿几乎猜不中，接受率被压到 **~48%**，decode 只有 ~49 tok/s。把 `draft_vocab.bin` 换成 SC117 自带的
**多语言词表（106299）**后，中文接受率升到 **~82–87%**，Swift decode **~49 → ~73（+48%）**，
base IQ3_S **~50.6 → ~65.8（+30%）**。**草稿头权重没动**，输出不变。

## 1. 为什么会这样

Strata 的 MTP（多 token 预测）草稿头有三部分：

| 文件 | 作用 | mtp-rt（en） | SC117 的 rt（multi） |
|---|---|---|---|
| `dense.bin` | 草稿层 dense 权重 | 116,099,072 B | 116,099,072 B（**sha256 相同**） |
| `experts.bin` | 草稿层专家权重 | 707,788,800 B | 707,788,800 B（**sha256 相同**） |
| `draft_vocab.bin` | 草稿下标 → 主词表 token id | **40,525 条** | **106,299 条** |

- 两边 `dense.bin` / `experts.bin` **逐字节相同** → 是**同一个草稿头**，不是「另一个模型」。
- 唯一区别是词表大小。`dense.txt` 元数据也相同（同一层结构）。
- 实测 `en == multi[:40525]` = **True**：**multi 就是 en 后面追加了更多 token**，前缀完全一致，无重排。
- 两者 token id 范围都是 0–248319（同一个 Qwen 主词表），multi 是 en 的**严格超集**。

**后果**：英文 prompt 时 en 词表够用（接受率 ~85–88%），中文 prompt 时 en 词表猜不中中文 token
（接受率 ~48–53%）→ decode 慢。multi 词表覆盖中文 token，接受率回升。

## 2. 为什么换它安全

1. **输出是验证过的**：草稿是推测式的，主模型会**逐个 verify**，所以**输出永远等于主模型的结果**。
   换草稿词表**不可能改变答案**，最坏只是变慢。
2. **词表是超集前缀**：`en == multi[:40525]`，前 40525 个草稿下标一一对应，multi 只多不少。
3. **权重相同**：连草稿头都不用换，只换 `draft_vocab.bin` 一个文件。
4. 实测输出正常（17×23=391、自我介绍、中译英、冒泡排序均正确）。

## 3. 实测（同一引擎，只换 `draft_vocab.bin`）

Swift IQ3_XXS（主配置 `--batch 2` + #910 栈）：

| prompt | en 词表 | **multi 词表** | 提升 |
|---|---:|---:|---:|
| 中文 decode | ~49.4 | **~73.0** | **+48%** |
| 中文接受率 | ~48% | **~87%** | — |
| 英文 decode | — | ~67 | — |

base IQ3_S（单并发，`strata-qwen-iq3_s.json`）：

| prompt | en 词表 | **multi 词表** | 提升 |
|---|---:|---:|---:|
| 中文 decode | ~50.6 | **~65.8** | **+30%** |
| 英文 decode | ~70 | ~68 | ~0 |

SC117 IQ3_S 本来就用自带 multi（`/home/likan/models/sc117-iq3_s/strata/rt`），无需改动。

## 4. 怎么换 / 怎么退

```sh
cd ~/strata
# 换（已做）：备份 en 词表，覆盖为 multi
cp -p mtp-rt/draft_vocab.bin mtp-rt/draft_vocab.bin.en.bak
cp -f /home/likan/models/sc117-iq3_s/strata/rt/draft_vocab.bin mtp-rt/draft_vocab.bin
# 重启（引擎在启动时读取）
./start_iq3.sh

# 退：恢复 en 词表
cp -f mtp-rt/draft_vocab.bin.en.bak mtp-rt/draft_vocab.bin
```

- 影响范围：所有 `--mtp /home/likan/strata/mtp-rt` 的配置（**Swift 主部署 + base IQ3_S**）。
- SC117 用自己的 rt，不受影响。
- 已备份：`mtp-rt/draft_vocab.bin.en.bak`（sha `3691…`）。

## 5. 教训

**推测解码的 tok/s = 每步接受的 token 数 × 步频**。之前评估 #910 时用 en 词表，中文接受率只有 ~48%，
每个 verify window 很短，掩盖了 #910 流水线重叠的收益（当时测出「无收益」）。换成 multi 后接受率 ~78–87%，
#910 的收益才显现（Swift 单并发 **+16%**，见 `V0139-MIGRATION.md`）。
**评估 decode 优化前，先确认草稿接受率是正常的。**
