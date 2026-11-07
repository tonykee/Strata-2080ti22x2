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

## 2026-11-07: CPU-lane enable attempt (not adopted)

Enabled `STRATA_NATIVE_EXPERTS`, changed the namespace alias to `strata::kernels::cpu`, added
`native_gu_rows_split` / `native_down_rows_split`, raised `kNativeActBytes` / `kNativeHBytes` to
8192 / 4096 (DeepSeek's 5120 / 2304), and allowed IQ1_S own rows for one token
(`DS4_IQ1S_ROWS=1`, `DS4_IQ_OWN=1`). It builds and runs, but:

| Config | Decode (128 tok) | 3.5K prefill | CPU lane |
|---|---:|---:|---:|
| CPU lane off (current best) | 27.71 tok/s | 185.5 tok/s | ~26 ms/token |
| CPU lane enabled | 17.70 tok/s | 39.8 tok/s | ~46 ms/token |

The Strata IQ1_S rows were not selected (the file is the pre-#695 scalar form; its one-token check/row
timing did not win), and the separate gate/up bases fell through to a scalar split path. This is worse
than the generic CPU lane, so the enablement was reverted. The branch keeps #5246 only.

A second attempt ported Maya #695's vectorized `iq1s_rows_avx2` (F16C, 16-block segments) into
`src/dsv41/ds4_iq_cpu.cpp`, plus the split-base declarations. It builds and runs, but:

| Pack | CPU lane | Decode 128 tok | 3.5K prefill |
|---|---|---:|---:|
| Maya-XS-v2, CPU lane enabled | ~46 ms/token | 17.70 | 39.8 |
| Maya-XS-v1, CPU lane enabled | ~58 ms/token | 14.57 | 75.9 |
| no CPU-lane enable (current best) | ~26 ms/token | **27.71** | **185.5** |

The own IQ1_S rows still were not selected by the dsv41 call path (no `CPU lane iq1_s rows` line);
the engine's `iq_cpu_fmt`/CPU-lane branch does not reach them in this pack, so the split-base path
dominates. This is not a drop-in port: it needs the newer `ds4_iq_cpu` + `ds4_cpu_prepare_rows` +
model call path from Maya, not just the row function. Reverted; branch keeps #5246 only.

Next step for this part: port the whole Maya CPU-lane path, not only #695's row kernel; that is the
1-2 day piece estimated earlier.

## 2026-11-07: full #695 CPU-lane port (works)

Replaced Strata's pre-#695 `ds4_iq_cpu.cpp` with Maya's post-#695 file (the vectorized IQ1_S rows),
added `ds4_iq512_kern.inc`, the split-base `native_gu_rows_split` / `native_down_rows_split`, and
raised `kNativeActBytes` / `kNativeHBytes` to 8192 / 4096. The CPU lane now reports:

```text
ds4: CPU lane iq1_s rows (4 tokens): native 515 us, ours 210 us -> ours
```

v2 pack, TP2, tuned flags:

| Config | Decode 128 tok | 3.5K prefill | CPU lane |
|---|---:|---:|---:|
| #5246 TP2 only | 27.71 tok/s | 185.5 tok/s | ~26 ms/token |
| **#5246 + #695 CPU lane** | **31.38 tok/s** | 185.8 tok/s | **21.3 ms/token** |

That is +13.2% decode from the CPU-lane port; the IQ1_S rows alone are 2.45x faster than the native
rows on this Xeon. Prefill is unchanged (the remaining prefill gap to Maya is the missing peer
all-reduce, not the CPU lane). Maya is still 42-49 / ~1000; cache substitution and peer prefill AR
are the next missing pieces.

## 2026-11-07: CUDA peer prefill all-reduce (opt-in, WIP)

Ported the two-rank peer-VRAM prefill transport to the CUDA build: `ar_peer_kernel` / `allreduce_peer` in
`ds4_kernels.cu`, `Group::peer_arb`, and a `DS4_HIP_PEER_AR=1` setup after expert placement. Each rank
allocates two receive slots (320 MiB/rank at the 8192-token chunk) plus flag/error blocks and enables
peer access in both directions. The chunk path uses it for all three per-layer all-reduces; the host
transport remains the default and is what an unset or `0` value uses.

First measured run, v2 pack, TP2, CPU lane:

| Config | 3.5K prefill | Decode |
|---|---:|---:|
| host prefill transport | 185.8 tok/s | 31.38 tok/s |
| `DS4_HIP_PEER_AR=1` | **196.2 tok/s** | 27.81 tok/s (noisy) |

The peer run's generated sequence differed from the host runs, and the decode number is not comparable:
the host was under heavy CPU/kcompactd load during the later runs (CPU lane 49-56 ms/token against 21 ms
in the earlier measurement), so decode timings and the dynamic expert-cache timing differ. `init_peer_ar`
now has `DS4_PEER_AR_SELFTEST=1`, which runs one 65536-float row through the host and peer transports and
compares every bit to the host-computed sum. Result: **host_ok=1 peer_ok=1 mismatches=0**. With the
expert cache disabled and dspark off, peer and host generated the same 8 tokens
(`455,2502,344,260,1606,10021,9162,943`), so the earlier sequence difference was cache timing, not the
peer arithmetic. Keep it opt-in; do not make it the default yet.

## 2026-11-08: cache-aware expert substitution (opt-in, WIP)

Ported Maya #821's `cache_sub::choose` / config and a CUDA route kernel. When `DS4_CACHE_SUB_RATIO=0.1`
(`DS4_CACHE_SUB_N=12`, `DS4_CACHE_SUB_RISK_BUDGET=0.6`), generation windows up to 16 tokens may replace
low-score non-strongest picks by resident alternatives; the resident test is the layer's VRAM table
(`vram_tab != 0`). Default is zero: native routing, no extra work. This is approximate and changes answers.

A/B on the current degraded host, v2, TP2, CPU lane, 128 tokens, dspark on, `--mmap-experts off`:

| Config | Decode | CPU lane |
|---|---:|---:|
| native | 18.41 tok/s | 44.07 ms/token |
| `DS4_CACHE_SUB_RATIO=0.1` | **34.49 tok/s** | **17.56 ms/token** |

Prefill was unchanged (39.9 vs 40.2 tok/s in that state). The generated 128-token sequence happened to
be the same on this repetitive prompt, but that is not a quality proof: other prompts and the 30-task
quality set still need to be run before any adoption. Keep it opt-in; not the default.

Note: the host is still in a degraded state after repeated 90 GB CPU-lane copies (CPU lane 44-96 ms/token
against 21 ms in the earlier clean-state 31.38 tok/s run). A host reboot is needed before the final
numbers are quoted.

## Launcher

`start_ds4_v2_2080ti.sh` is the 2x2080Ti launcher. It defaults to the measured configuration
(TP2, `--experts 4`, 8192 chunks, Engram 2048, CPU IV1_S own rows, peer AR on where available) and
leaves cache substitution off. Examples:

```sh
./start_ds4_v2_2080ti.sh --serve
./start_ds4_v2_2080ti.sh --ids 1,2,3 --n 32
DS4_CACHE_SUB_RATIO=0.1 ./start_ds4_v2_2080ti.sh --serve   # approximate; changes answers
DS4_HIP_PEER_AR=0 ./start_ds4_v2_2080ti.sh --serve          # host prefill transport
```

`DS4_HIP_PEER_AR=1` is the script default because it is bitwise checked; the engine still falls back
to the host transport when peer access or the setup is unavailable.

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
