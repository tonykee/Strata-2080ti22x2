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

The local config (ignored by git) is `strata-swift-latest.json`; it points at
`/home/likan/strata-latest/engine/strata` and `engine/strata-vision`.

```bash
cd /home/likan/strata-latest
./.venv/bin/python serve/server.py --engine strata \
  --config strata-swift-latest.json --port 8000
```

Do not switch `~/strata`'s production config to this engine until the image path and the full
SC117/Swift conversation tests have run.
