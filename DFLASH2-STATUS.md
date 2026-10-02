# DFlash2 现状与可行性（Qwen3.8-Flash-Next on Strata）

> 记录时间：2026-10-02。用于跟踪上游进度、判断对 `~/strata-v0134`（双 2080 Ti 22G）
> 是否有意义。**结论：上游有计划，但处在最早的 RFC/可行性阶段；模型侧缺 Flash-Next 的
> DFlash2 草稿 checkpoint，是最大 blocker。**

## 1. DFlash2 是什么

- **block-diffusion 草稿模型**（不是独立语言模型），用于投机解码：
  - 传统草稿（MTP/EAGLE-3）**串行**逐 token 生成，草稿成本随长度线性增长；
  - DFlash **一次并行**生成整个 block（anchor + N 个 MASK 位置扩散去噪），草稿成本几乎与 block 大小无关；
  - **DFlash2** 在 DFlash 基础上加了 *two-tap in-block convolution* 和 *learned pairwise path selector*（在每位置 top-k 里选一条连贯路径）。
- 验证仍由目标模型做，**lossless**（不改变输出分布，只改速度）。
- 典型配置：block/`num_speculative_tokens` ≈ 7；实测要自己扫（N=7 常是高点）。

## 2. 上游 Strata 的计划（关键）

| 项 | 状态 |
|---|---|
| **Issue #347**「DFlash2 support for Qwen3.8-Flash-Next」 | j-luwierski，2026-10-01，**已关闭**（转为 PR） |
| **PR #366**「[RFC] EXPERIMENTAL Investigate DFlash2 support for Qwen3.8-Flash-Next」 | **Draft**，1 commit，无 review / assignee / milestone |

链接：
- https://github.com/Niko1221/Strata/issues/347
- https://github.com/Niko1221/Strata/pull/366

PR #366 明确声明：**不声称有兼容 checkpoint、不声称实现完成、不声称比 MTP 快**；负结果也算结论。它把工作拆成三问：

1. **模型可用性**：有没有/能不能拿到与 Flash-Next 兼容的 DFlash2 drafter？
2. **运行时正确性**：Strata 能否暴露所需 target features，并正确 verify/commit/rollback draft 块？
3. **实际性能**：把 drafting、verification、显存压力、专家缓存变化都算进去后，是否真的更快？

**PR 原文要点（对你这台尤其重要）：**
- 公开的 DFlash2 checkpoint 只有 **Qwen3.8-27B**（`incoai/Qwen3.8-27B-DFlash2`、`z-lab/...`），
  **不能当作 Flash-Next 的兼容 checkpoint**；Flash-Next 的 MTP 权重**不是** DFlash2 权重。
- 需要一个**针对 Flash-Next 训练**的 drafter（体量、quant 配置、tokenizer、chat template 都要对齐）。
- **显存预算**：多一个 drafter 会占显存 → 压缩专家缓存 → **可能整体更慢**（对双 2080Ti + 256K 是关键风险）。
- 非目标（首个里程碑不做）：多卡、并发、多模态、长上下文、与 MTP 组合。
- 首个小实验是 mock 的 2-token block 验证 verifier/rollback，**不需要真 drafter**。

## 3. DFlash2 生态与收益（不是 Flash-Next）

已支持：vLLM（`"method":"dflash"`）、SGLang Spec V2、llama.cpp（`--spec-type draft-dflash` + `-hfd`）、MLX、NVIDIA NeMo / SGLang **SpecForge**（训练）。

参考收益（不同目标模型，**均非 125B Flash-Next**）：
- Qwen3.8-**27B**：DFlash2 相对 baseline **~3.4x**；相对该模型自带 **MTP ~1.3–1.5x**。
- Qwen3.6-35B-A3B FP8：baseline 207 → DFlash2 **484 tok/s（2.3x）**，c=3 聚合 2.5x。
- 结论：DFlash2 主要赢在「**比 MTP 并行**」，收益量级取决于接受长度和验证成本。

## 4. 对你（双 2080 Ti 22G + Flash-Next IQ3_XXS）的现实结论

- **现在不可用**：Strata 未实现，且**没有 Flash-Next 的 DFlash2 草稿**。
- 你旧配置里的 `qwen27b-fp4-dflash2-7-...` 是配 **27B** 的，不是 Flash-Next。
- 想立刻体验 DFlash2：只能换栈到 **vLLM / llama.cpp + Qwen3.8-27B**（有现成 drafter）——
  那是另一个模型（质量、显存、上下文都不同）。
- 等 Strata 支持的前提是：先有人给 Flash-Next 训出 drafter（见 §5），再在引擎里实现 block 验证/回滚；
  PR 目前只是调查。**不要把它当近期的解码提速手段**。

## 5. 用 SpecForge 训练 Flash-Next DFlash2 drafter 的可行性（评估）

> 若没有现成 checkpoint，PR #366 给了训练路线：用 **SpecForge** 的 DFlash2 训练
> （`training.strategy: dflash` + `DFlash2DraftModel`，支持 `data.hidden_states_path` 离线训练，
> 目标模型训练时不必常驻）。

### 需要的步骤
1. **确定 Flash-Next 的 feature 契约**：选 3 个 target 层（early/mid/late），记录
   归一化/残差前后、shape、token 对齐（对齐错一个 token，loss 会降但 drafter 无用）。
2. **给 Strata 加 feature 导出**：两遍法——先用目标模型生成 token 序列，再把整段 replay/prefill
   抓 hidden states（比逐 token dump 便宜）。先验证 replay vs decode 的特征等价。
3. **小数据集门禁**：16–32 条短序列**过拟合** → 查 future-token 泄漏 → 再 ~1M token pilot。
4. **导出 checkpoint** → 参考实现正向验证 → Strata 集成 → verifier/rollback → 三方基准。

### 成本/风险（针对本机）
- **数据量**：3 个 tap × hidden 2560 × bf16 ≈ **15 KB/token** → 1M token ≈ **~15 GB**，本机可抓。
  抓特征本身用现有 2×2080Ti 跑目标模型即可（慢但可行）。
- **训练算力**：drafter 只有 ~5 层，比目标小得多；但 SpecForge/NeMo 训练栈通常依赖
  **bf16 + FlashAttention-2**，而 **Turing sm_75 支持有限**——这是本机最大的不确定性，
  可能需要在云端 A100/H100 上训练（PR 建议 ~30B 目标用 4×H100）。
- **词表投影**：Flash-Next vocab 248320，decoder 投影是主要显存开销，需实测。
- **最终收益未知**：PR 自己说，drafter 占的显存会压缩专家缓存，双 2080Ti + 256K 下**可能被抵消**。

### 结论
- **值得跟踪，不值得现在投入**：blocker 不在引擎，而在「Flash-Next 的 DFlash2 草稿」，
  且训练栈对 sm_75 不友好、最终收益在本机可能被显存挤掉。
- 上游若把 PR #366 推到能用，再评估移植；在那之前，当前 MTP + suffix-draft 就是本机的现实上限。

## 6. 跟踪入口

- Strata PR #366 / Issue #347（见 §2）
- DFlash 参考实现：https://github.com/z-lab/dflash
- DFlash2 说明：https://inco.ai/blog/dflash2/
- 27B 草稿（仅参考，不兼容 Flash-Next）：https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
- SpecForge 训练：https://github.com/sgl-project/SpecForge
