# ds4-maya-merge: merging Strata and Maya DS4.1 strengths

Branch from `v0.2.11-2080ti` for porting Maya's DeepSeek optimizations into Strata's
`strata-dsv41`, without disturbing the Qwen production branch.

## 2026-11-07: PR #5246 (capacity-aware block-aligned GGUF TP)

Cherry-picked `287cfed08` on this branch. It builds for sm_75. TP2 is now selected instead of
falling back to the layer split:

```text
ds4: GGUF TP balance group 0: moved 7 blocks (256 values), avg blocks/layer CUDA0=4.33 CUDA1=4.67
ds4: CUDA0 layers [0, 42] (tensor parallel 2, part 0) + head
ds4: CUDA1 layers [0, 39] (tensor parallel 2, part 1)
```

Measured with `Maya-XS-v2/maya-ds4-v2`, `--tp 2 --dspark on --experts 4 --mmap-experts off
--reserve-mib 768 --prefill-chunk 8192`, Engram cache 2 GB, stream/land on:

| Config | Decode (128 tok) | 3.5K prefill |
|---|---:|---:|
| no TP2 (layer split, same tuned flags) | 25.96 tok/s | 221.8 tok/s |
| **PR #5246 TP2** | **27.71 tok/s (+6.7%)** | 185.5 tok/s (-16.4%) |

The PR's +52% headline is from 4x RTX 4090 where all 15,744 experts fit in VRAM. Here the CPU lane
still has 12,670 experts (90.7 GB) on one NUMA part, so TP2 improves decode only slightly and costs
prefill (host all-reduce, no NVLink peer path yet). CUDA1 finished with 4,622 MiB free, so the
capacity balance is not filling the second card completely.

## Next, in order

1. **CPU IQ1_S lane** — Strata's `ds4_iq_cpu.cpp` has the vectorized IQ1_S rows but
   `#undef STRATA_NATIVE_EXPERTS` compiles them out; the Maya split-base CPU rows
   (`native_gu_rows_split` / `native_down_rows_split`) are not in Strata. This is the largest
   remaining decode lever.
2. **PR #4302** — second card as a DeepSeek helper expert cache (open, conflicting), or
   `DS4_TP_SHARES`/`--vram-experts` tuning to use CUDA1's 4.6 GiB.
3. **Maya peer prefill all-reduce** — for prefill; the current TP2 uses the host path.
4. **PR #4619 `DS4_LOOKUP`** — repeated/copying agent traffic, +74% in its report; clean PR.
5. **Cache substitution** — Maya's last +38%, but changes outputs; quality gate required.

Raw runs are in `/tmp/strata-5137/` on the build box.
