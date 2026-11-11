# Mainline 0.2.13 / engine 0.1.59 merged into the DS4 branch (2026-11-11)

Status: **local merge result, built and benchmarked on the 2x RTX 2080 Ti box.**  The source branch is
`merge-main-0.2.13`; it is merged into `ds4-cuda-prefill` in the same push.  The pre-merge branch is
kept by the tag `backup-pre-main-merge-20261111`.

## What was merged

- Upstream: `origin/main` = `dbb2d168c`, **0.2.13 / engine 0.1.59** (401 commits ahead of the old base,
  351 non-merge; 1,068 files, +207k / -38k).
- Local: `ds4-cuda-prefill` had 118 commits (35 DS4 commits, ~3,274 lines in `src/dsv41`, plus the
  local 1 GB arena, conv-cache root, `/v1/models` context fields and Swift tuning).
- Merge commit `25d9ce07b`; follow-ups `303fe8f75` (DS4 declarations) and `5abc413ba` (48 GB helper rule
  and wrapper default).
- Conflicts: only two:
  - `CMakeLists.txt` — upstream moved the build graph into `cmake/*.cmake`; took upstream and added the
    local `strata_spec` link to the DS4 target in `cmake/strata_gpu_engine.cmake`;
  - `include/strata/kernels/cpu/native_expert.hpp` — took upstream's geometry constants and restored the
    local `native_gu_rows_split` / `native_down_rows_split` declarations.
- The DS4 target needed one extra fix after the merge: those two split-row declarations were missing from
  the merged header; `build-merge/strata-dsv41` then compiled.

Local features that survived: DS4 engine (35 commits), `STRATA_ARENA_HUGE_1G`, `STRATA_Q36_HUGE_1G`,
conversation-cache root overloads, `/v1/models` `max_model_len` / `context_length`.

## Build

```sh
cmake -S . -B build-merge -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_DSV41=OFF
cmake --build build-merge --target strata -j 16

cmake -S . -B build-merge -DSTRATA_BUILD_DSV41=ON
cmake --build build-merge --target strata-dsv41 -j 16
```

Result:

```text
build-merge/strata        0.1.59.1
build-merge/strata-dsv41  0.1.59.1
```

The text engine builds with CUDA 12.8 / sm_75.  The DS4 target builds after the split-row declaration
fix; it has not been benchmarked in this pass.

## Swift A/B (same fast config, no huge-page pool, THP)

| Test | 0.1.57 fast text | merge 0.1.59.1 fast text | Change |
|---|---:|---:|---:|
| chat decode median | 82.0 | 81.3 | -0.9% (noise) |
| 4K prefill | 1,239.4 | 1,244.0 | +0.4% |
| **16K prefill** | 1,674.0 | **1,824.4** | **+9.0%** |
| agent decode median | 75.7 | **82.0** | **+8.3%** |
| 50K prefill / decode | 1,880.0 / 82.8 | 1,893.1 / 82.0 | +0.7% / -1.0% |
| 100K prefill / decode | 2,047.5 / 114.7 | 2,073.8 / 118.0 | +1.3% / +2.9% |

The 16K/20K and agent gains are the real change; 4K/50K are flat; 100K is slightly better.  Data:
`data/quick-0.1.59-thp.json`, `data/x99-0.1.59-thp.json`.

## 48 GB 2 MB huge pages (the right size)

The expert arena is 46.84 GiB = 23,983 2 MB pages.  A 48 GB pool (24,576 pages) leaves 593 pages free and
does **not** squeeze the page cache.  96 GB (the old passwordless helper) did squeeze it: on the fast
config the 4K prompt fell from 1,239 to about 1,005 tok/s.  48 GB recovers it.

`start_swift_2mb_cache.sh` now defaults to 48 GB and uses the narrow passwordless rule
`packaging/99-strata-hugepages.sudoers`:

```sudoers
likan ALL=(root) NOPASSWD: /usr/local/sbin/strata-ds4-hugepages 48
```

One-time install (root):

```sh
sudo install -m 0440 -o root -g root packaging/99-strata-hugepages.sudoers \
  /etc/sudoers.d/99-strata-hugepages
sudo visudo -cf /etc/sudoers.d/99-strata-hugepages
```

The wrapper reserved the pool itself and the engine logged:

```text
STRATA_ARENA_HUGE_1G: no 1 GB pool (needed 47 pages, nr_hugepages=0); trying 2 MB pages
expert arena: ... hugetlb 2 MB pages
HugePages_Total=24576  HugePages_Free=593
```

The wrapper config in this run was `strata-swift-2mb-cache.json` with `exe=build-merge/strata`:
single stream (batch 1), `--vram-reserve-mib 700`, vision on.  The engine is 0.1.59.1; the vision encoder
is the older 0.1.57 `engine/strata-vision` and works with it.

### Quick suite, 48 GB, single stream + vision

| Test | merge THP fast text | **2 MB / 48 GB, single + vision** |
|---|---:|---:|
| chat decode median | 81.3 | **82.9** |
| 4K prefill | 1,244.0 | **1,224.9** |
| 16K prefill | 1,824.4 | **1,813.8** |
| agent decode median | 82.0 | **87.5** |
| agent first token | 3.82 s | **2.97 s** |

The 4K/16K are within 1.5% of the no-pool fast config; the agent is better.  The old 96 GB pool's
4K 1,005 / 16K 1,610 are gone.

### 50K / 100K, 48 GB, single stream + vision

| Test | merge THP fast | **2 MB / 48 GB single** | old 128 GB 1DPC record |
|---|---:|---:|---:|
| 50K prefill | 1,893.1 | **2,219.4** | 2,182.7 |
| 50K decode | 82.0 | **107.9** | 128.2 |
| 100K prefill | 2,073.8 | **2,152.3** | 2,378.9 |
| 100K decode | 118.0 | **129.4** | 125.8 |

At 50K the 48 GB configuration is slightly **above** the old 128 GB 1DPC prefill record; 100K decode is
above it; 100K prefill is still about 9.5% below; 50K decode is the first long run (cache 96.5%) and is
below the old 128.2, while the warm 100K run reached 129.4.

Data: `data/quick-0.1.59-2mb48.json`, `data/x99-0.1.59-2mb48.json`.

## How to run

```sh
# 48 GB pool, single stream (batch off), vision, merge engine
./start_swift_2mb_cache.sh

# two requests decode together (batch 2), same 48 GB pool
./start_swift_2mb_cache.sh -b2

# no pool, fast/adapt text-only (the other config)
./start_swift_fast.sh --no-vision
```

The wrapper reads `strata-swift-2mb-cache*.json` (gitignored).  In this test its `exe` points to
`build-merge/strata`; the old `engine/strata` 0.1.57 remains for rollback.  The wrapper reserves the pool
only when it is not already big enough, and releases the pool it reserved on Ctrl+C/SIGTERM.
`stop_swift_2mb_cache.sh` handles a killed wrapper.

## Caveats

- The merge branch is **not** the same as upstream main: local DS4/1 GB/conv-cache/models changes are
  carried on top of 0.2.13.  Upstream did not cover those (see the earlier comparison).
- The DS4 target compiled, but no DS4 warm-up or 6-turn benchmark was run on this merge yet.
- Vision uses the 0.1.57 encoder.  A 0.1.59 `strata-vision` build has not been made; the 0.1.57 encoder
  answered a red-square image with "Red" against the 0.1.59.1 engine.
- The 1 GB page path is still local (`STRATA_ARENA_HUGE_1G` / `STRATA_Q36_HUGE_1G`); the 1 GB pool is 0,
  so this run used 2 MB pages.
- The current server at the time of writing: wrapper PID 353245, server PID 353299, 0.0.0.0:8000,
  `llama_local`, 48 GB pool, single stream + vision.
