# LOCAL 2080 Ti fork — base 0.2.11 / engine 0.1.57

This branch is the 2026-11-07 refresh of the 2080 Ti deployment: upstream `v0.2.11` (engine 0.1.57)
plus only the local changes that are still needed.  The old `v0.1.45-2080ti` tree stays untouched
until this branch passes the same validation.

## Local deltas kept

1. **Conversation-cache root at a named index** — `include/strata/program/conv_cache.hpp` and
   `src/program/generate.cpp`.  A system prompt that ends past the first periodic chunk end leaves grid
   checkpoints ahead of its own checkpoint (#2228, #3090); the retention policy now pins the caller's
   root instead of always item 0.  `conv_cache_test` passes; the new overloads are not yet covered by
   the upstream test file (the old tree's added cases were not copied).
2. **`/v1/models` context names** — `serve/server.py` now reports `max_model_len` and `context_length`
   beside `meta.n_ctx`, so vLLM/OpenRouter-style clients size their window correctly.
3. **Pipeline windows beside batch slots** — no code patch: the config sets
   `STRATA_PW_WITH_BATCH=1`, upstream 0.1.57's supported switch, which keeps the solo path pipelined
   while batch windows run without it.

## Not ported yet

- The SYCL copy of the root lookup (`sycl/src/program/generate.cpp`); this branch is a CUDA build.
- The old tree's extra `conv_cache_test.cpp` cases and the one-line `/v1/models` server test are
  pending; the code builds and the existing test passes.

## Decode-step optimization found on 2026-11-07

`STRATA_EXP_MODE=8` (the video's mode 8 expert layout) is a real win on sm_75 here, even though
0.1.57's default for a Turing card is mode 4.  Swift 384K/int8, temperature 0, thinking off, 5 warm
runs, `STRATA_DECODE_TIMING=1`:

| Config | short code | short list | 100K decode |
|---|---:|---:|---:|
| production-like (batch 2, suffix-draft 8, spec-min-p 0.65) | 110.3 tok/s / 32.8 ms | 88.4 / 26.4 ms | 82.6 tok/s |
| + `STRATA_EXP_MODE=8` | **117.9 / ~29 ms** | **97.3 / ~26 ms** | **99.4** |
| peak profile: no batch, suffix-draft 0, spec-min-p 0.5, mtp-hnorm stream, + mode 8 | **126.1** | **102.9** | 90.4 |
| peak profile + layer-split 22 | 127.4 | 100.3 | 86.6 |

- `STRATA_EXP_MODE=8` is the one to keep in the daily config: it helps short and 100K, and keeps
  batch/suffix-draft.
- The peak profile is only for single-task short peaks; at 100K the suffix draft's acceptance is
  worth more than its overhead (90.4 against 99.4 for the production-like config).
- Layer-split 22 is not a win on this box (24/24 stays).
- The local configs already carry `"STRATA_EXP_MODE": "8"`.

**Quality check (2026-11-07, [full results](bench/results/2026-11-07-exp8-quality/README.md)).**
Eight greedy prompts with mode 4 and mode 8, thinking off, same config: math, logic, two Python
functions (compiled and tested), JSON, Chinese, EN→ZH and a 20K-token needle. **Both modes passed
8/8; all eight responses were byte-for-byte identical text**, including the 20K needle and the
generated code. This is a task-level smoke test, not a logit/KL or bitwise proof.

### Other switches tried on 2026-11-07 (not adopted)

All on top of the daily config + mode 8; same short-code/list/100K runs:

| Switch | Short code | Short list | 100K prefill / decode | Verdict |
|---|---:|---:|---:|---|
| `STRATA_MTP_STAGE0=1` | 117.8 | 101.3 | 2537 / **76.7** | short list +4%, long −23%: no |
| `STRATA_PA_V2T=1` + `STRATA_SPLIT_MTP_BATCH=1` | 117.9 | 97.3 | **2668** / 87.5 | prefill +4.4%, decode lower: no |
| `STRATA_ATTN_DAILY=1` + `ATTN_MERGE_V2=1` + `EXCHANGE_ROTATE=1` + `ADAPT_EVICT_SYNC=1` + `ADAPT_LAG=1` | 118.1 | 97.5 | 2379 / **70.6** | short unchanged, long much worse: no |
| `STRATA_GDN_TC=2` (port of #4876) | 118.0 | 97.3 | 2275 / **64.0** | no gain, long worse: no |
| `STRATA_PIPELINE_SNAP_RING=1` (port of #3944) | 122.8 | 97.6 | 2538 / **89.4** | short code +4%, long −10%: no |
| #4549 draft-head room (no trim change) | 117.9 | 97.3 | 2531 / **86.4** | cache slots unchanged, no gain: no |
| #5002 `STRATA_DP_PREFILL=1` (no batch, no trim, `STRATA_KV_HOST_DMA=1`) | — | — | **1623** / 81.2 | DP linked, but 100K prefill −36% vs 2555: no |
| `--layer-split 22` (24/24 stays) | 127.4 peak | 100.3 | 86.6 | only peak short code +1%: no |

Single runs each; the long-context differences are large enough (10–35%) that thermal noise does
not explain them. The daily profile keeps only `STRATA_EXP_MODE=8`; the peak profile additionally
drops batch/suffix for short single-task work.

## Validation on the 2080 Ti box (X99, 2x RTX 2080 Ti 22G, 220 W)

- Built `strata` and `strata-vision` with CUDA 12.8, `sm_75`, Release.
- Swift 1.5 IQ3_S, 384K/yarn 1.5, `--kv int8`, 20K/50K/100K, 512-token decode:
  no regression against the unmodified 0.1.57 binary measured at the same thermal state
  (prefill identical within ~1%; decode within run-to-run noise).  Against 0.1.45 the result is
  also a tie at equal temperature; the early 0.1.57 run that looked much faster started with GPU0
  at 41 °C against 55–75 °C in the other runs, so it is not claimed as a speedup.
- SC117 starts on 0.1.57 (80 GB pack, same 384K/int8 config) and reports `max_model_len=393216`;
  its throughput was not benchmarked in this pass.
- `/v1/models` returns `max_model_len` / `context_length`; the vision encoder builds.

## Run it

The local configs (ignored by git) are `strata-swift-latest.json` and
`strata-sc117-latest.json`; they point at `/home/likan/strata-latest/engine/strata` and
`engine/strata-vision`.

```bash
cd /home/likan/strata-latest
./start_iqs3_swift.sh     # Swift 1.5 IQ3_S
./start_iq3s_117.sh       # SC117 abliterated IQ3_S
./stop.sh                 # stops this branch only, not ~/strata
```

For the 256 GB / 2DPC box, the huge-page wrapper reserves a 2 MB pool before the server and
releases it on exit:

```bash
./start_swift_2mb_cache.sh            # default: no concurrency (batch off), 96 GB pool
./start_swift_2mb_cache.sh -b2        # two requests decode together (batch 2)
HUGEPAGE_GB=48 ./start_swift_2mb_cache.sh -b2   # 48 GB is enough; asks for sudo
./start_swift_2mb_cache.sh --check    # reserve, print, release; does not start a server
./stop_swift_2mb_cache.sh             # only if the start script was killed with SIGKILL
```

The wrapper refuses to start when port 8000 or another `strata` engine is already in use.  It
releases only a pool it reserved; a pool reserved by DS4 or by hand is left alone.

The start scripts refuse to run when port 8000 is already in use, so stop the production tree
(`/home/likan/strata/stop.sh`) first if it is running.

Do not switch `~/strata`'s production config to this engine until the image path and the full
SC117/Swift conversation tests have run.

## Swift on 256 GB (2026-11-10)

The 256 GB 8x32 GB 2DPC upgrade costs Swift prefill about 16-18% against the 128 GB 4x32 GB 1DPC
record (50K 2,182.7 -> 1,813.7, 100K 2,378.9 -> 1,977.9 tok/s).  The same-config 0.1.45 control on
256 GB is 1,826.9/1,996.8, so this is the memory state, not the 0.1.57 code (0.1.57 decode is 3-4%
above 0.1.45 at 50K/100K).

2 MB hugepages for the expert arena, `--vram-reserve-mib 700` and `--batch 1` recover about 7-9%
(50K 1,934-1,979, 100K 2,111-2,133) and raise 100K decode to 131-134 tok/s.  `--prefill 4096`,
`--pool-workers 12`, removing `STRATA_EXP_MODE=8`/`STRATA_PW_WITH_BATCH`, and `--pipeline-windows 0`
all made it worse.  Full numbers, the rejected settings and the harness:
[bench/results/2026-11-10-swift-2dpc-prefill/README.md](bench/results/2026-11-10-swift-2dpc-prefill/README.md).
The remaining 9-11% needs 1DPC (or a future 1 GB-page/PLE-hugepage experiment); the BIOS has no
rank-interleaving or command-rate setting.

