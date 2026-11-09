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

A/B after the host settled, v2, TP2, CPU lane, 3.5K prompt, 128 tokens, dspark on,
`--mmap-experts off`, peer AR on:

| Config | Prefill | Decode | CPU lane |
|---|---:|---:|---:|
| cache sub off (default) | 196.3 tok/s | 38.83 tok/s | 15.18 ms/token |
| `DS4_CACHE_SUB_RATIO=0.1` | **197.0 tok/s** | **49.23 tok/s** | **9.65 ms/token** |

That is +26.8% decode at the same state; prefill is unchanged. The generated 128-token sequence happened
to be the same on this repetitive prompt, but that is not a quality proof: other prompts and the 30-task
quality set still need to be run before any adoption. Keep it opt-in; not the default.

For reference, the same branch before peer AR/cache sub measured 31.38 tok/s decode / 185.8 prefill.
The earlier 18-34 tok/s cache-sub A/B was taken while `kcompactd` was still cleaning up after repeated
90 GB CPU-lane copies (CPU lane 44-96 ms/token); do not use those numbers.

## Memory state: why a reboot was needed, and the fix

The 90.7 GB `--mmap-experts off` copy is anonymous `MADV_HUGEPAGE` memory.  Repeated load/free
cycles can leave the host fragmented; the next copy falls back to 4 KB pages and the CPU lane goes
from about **10-15 ms/token** to **40-96 ms/token**.  That is a host-memory problem, not a model
reload requirement, and a reboot is not the right fix.

The engine already has an explicit `MAP_HUGETLB` arena (`DS4_COPY_HUGETLB`).  Reserve the 2 MB pool
once, and every process reload reuses the same pages; no compaction or reboot between loads:

```sh
sudo tools/ds4_hugepages.sh 96 --persist
DS4_COPY_HUGETLB=1 ./start_ds4_v2_2080ti.sh --serve
```

The default 96 GB leaves about 28 GB for the OS and page cache on this 127 GB host; the v2 CPU-lane
copy is about 90.7 GB.  Verify before launch:

```sh
grep -E 'HugePages_(Total|Free)|Hugepagesize' /proc/meminfo
```

`HugePages_Free` must cover the copy; during a load it drops by the copied bytes and returns after
exit.  If the pool cannot be reserved, the alternatives are `--mmap-experts on` (no 90 GB anonymous
copy; reads in place), a smaller `DS4_COPY_GB` hybrid, or keeping one server process alive.  The
huge-page path is the one that keeps the measured CPU-lane speed across reloads.

## Scoped huge-page pool: clean on exit, no assumption about the next run

A process exit only returns `MAP_HUGETLB` pages to the kernel's huge-page pool; it does not return
the pool to ordinary memory (`HugePages_Total` stays).  That is why a persistent `vm.nr_hugepages`
reservation can look like it survived the engine.  The pool is not model data, but a later Qwen load
sees less ordinary RAM.

To make the pool process-scoped, use the service unit in `tools/strata-ds4-2080ti.service`:

```sh
sudo cp tools/strata-ds4-2080ti.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now strata-ds4-2080ti
```

- `ExecStartPre` (root) reserves 96 GB.
- `ExecStart` runs the engine as the `likan` user.
- `ExecStopPost` (root) runs `tools/ds4_hugepages_release.sh`: `vm.nr_hugepages=0`, then
  `/proc/sys/vm/compact_memory`.

So `systemctl stop strata-ds4-2080ti` returns the pool to ordinary memory and compacts, regardless
of whether DS4, Qwen, or something else starts next.  The same is available manually:

```sh
sudo tools/ds4_hugepages_release.sh
```

Caveat: release+compact is not identical to a reboot for every kernel cache/allocator detail; a
reboot is still the only absolute reset.  It does remove the huge-page pool deterministically.  For
a wrapper instead of systemd, run the engine as a child and call the release script from `trap EXIT`;
`SIGKILL` of the wrapper bypasses traps, while systemd's `ExecStopPost` handles service stops.

If the pool is not wanted at all, `--mmap-experts on` (or a small `DS4_COPY_GB`) avoids the 90 GB
anonymous copy and has no global pool to clean; process exit then returns its memory normally.

## CPU-lane start/stop

`start_cpu_lane.sh` and `stop_cpu_lane.sh` are the process-scoped pair:

```sh
./start_cpu_lane.sh                 # reserves 96 GB huge pages (sudo), starts the server as you
./stop_cpu_lane.sh                  # stops server/engine, then releases the pool (sudo)
DS4_HUGEPAGE_GB=0 ./start_cpu_lane.sh   # no sudo at all; ordinary THP arena
```

The Python server and `strata-dsv41` engine run as the invoking user.  Only
`vm.nr_hugepages` (reserve/release) and `compact_memory` need root; the scripts use `sudo` for
exactly those steps.  `DS4_HUGEPAGE_GB=0` skips both, so start/stop need no admin at all; the
engine then uses its ordinary THP arena.

The server is the normal `serve/server.py`; its config is `tools/strata-ds4-v2-2080ti.json` (arch
`deepseek41`, port 8000, TP2, `--mmap-experts off`, 4 experts, dspark on, peer AR on, cache
substitution off).  The start script writes `.strata-ds4-v2.pid`; the stop script only matches this
tree's server and engine paths, not production `/home/likan/strata`.

## Clearing the host state after a load

After the process exits, the CPU-lane arena is already unmapped; what can remain is allocator
fragmentation (THP/order blocks).  `drop_caches` is not the fix for that: it drops file pages, not
free-page fragmentation.  The useful command is compaction:

```sh
sudo tools/ds4_compact_memory.sh
# equivalent:
sudo sh -c 'sync; echo 1 > /proc/sys/vm/compact_memory'
```

If a huge-page pool was reserved, release it first (that also compacts):

```sh
sudo tools/ds4_hugepages_release.sh
```

`stop_cpu_lane.sh` does the release automatically when a pool exists.  For an ordinary-THP run
(`DS4_HUGEPAGE_GB=0`), set `DS4_COMPACT_ON_STOP=1` to make stop compact too:

```sh
DS4_HUGEPAGE_GB=0 DS4_COMPACT_ON_STOP=1 ./stop_cpu_lane.sh
```

`compact_memory` is best effort: movable user pages (the 90 GB arena) are coalesced back into
higher-order blocks, but unmovable kernel allocations can still prevent a perfect post-reboot
layout.  A reboot remains the only absolute reset; compaction normally recovers enough for the next
THP allocation.  `drop_caches` is a separate, optional operation (`sync; echo 3 > /proc/sys/vm/drop_caches`)
and mainly makes the next load colder.

## `DS4_COPY_GB=32` measured

`DS4_COPY_GB=32` is a static hybrid: the profile's hottest experts are copied to RAM, the rest stay
in the model file.  It **does not disable the automatic RAM store**; without `DS4_STORE_GB=0` the
store took another 66.8 GB and prefill fell to 85 tok/s.  The pure hybrid is:

```sh
DS4_COPY_GB=32 DS4_STORE_GB=0 DS4_COPY_HUGETLB=0 ...
```

Measured on the current host, 3.5K prompt, 128 tokens, dspark on, peer AR on, CPU lane own rows:

| Config | Prefill | Decode | CPU lane |
|---|---:|---:|---:|
| full 90.7 GB, cache sub off | 196.3 tok/s | 38.83 tok/s | 15.18 ms/token |
| full 90.7 GB, `DS4_CACHE_SUB_RATIO=0.1` | **197.0** | **49.23** | **9.65** |
| 32 GB hybrid, cache sub off | 172.0 | 35.87 | 17.49 |
| 32 GB hybrid, cache sub 0.1 | 175.7 | 34.25 | 17.31 |
| 32 GB hybrid **without** `DS4_STORE_GB=0` | 85.4 | 36.27 | 17.14 |

So 32 GB is a **memory-clean compromise**, not the fastest mode: about 8% slower decode and 12%
lower prefill than the full copy, but no 96 GB huge-page pool and much less anonymous THP pressure.
It still uses a 32 GB anonymous THP arena, so it is not completely free of fragmentation risk.
With a 256 GB host, the better default remains the full copy plus a scoped 96 GB huge-page pool;
`--mmap-experts on`/32 GB hybrid are the no-sudo fallbacks.

## Giving `likan` only the huge-page rights

Do not put the password in a script and do not grant general sudo.  Install the two helpers as
root-owned files and allow exactly those commands:

```sh
sudo tools/install_partial_admin.sh
```

The rule is:

```sudoers
likan ALL=(root) NOPASSWD: /usr/local/sbin/strata-ds4-hugepages *
likan ALL=(root) NOPASSWD: /usr/local/sbin/strata-ds4-hugepages-release
```

- The helpers are `root:root 0755` under `/usr/local/sbin`; `likan` cannot edit them.
- The reservation helper accepts its GB argument (1-192), so changing
  `DS4_HUGEPAGE_GB` does not require a sudoers edit; the helper itself rejects out-of-range values.
- Only "reserve N GB" (1-192) and "release the pool" are passwordless.
- No general `sudo`, no arbitrary `sysctl`, no shell.
- `visudo -cf` validates the rule before it is kept.
- `start_cpu_lane.sh` / `stop_cpu_lane.sh` prefer the installed helpers automatically.

If the helpers stay in the user-writable repository, giving them NOPASSWD would be equivalent to
granting root (the user could edit the script), so install them with this script instead.

## 2026-11-08: prefill streaming fix (13K 102 -> 451 tok/s)

Two bugs held the prompt path back:

1. `copy_ahead` always read every RAM-copied expert into pinned staging, even when the copy
   itself was already pinned.  The prompt then moved each expert RAM -> staging -> GPU.  With
   `DS4_COPY_HUGETLB=1` (arena on 2 MB pages) and the automatic `DS4_COPY_PIN`, the direct
   DMA path is now used; no staging is needed for a pinned copy.
2. `DS4_PROF` had a 6-entry local sample array while the chunk profile uses stages 0-10; it
   overwrote the stack (SIGSEGV).  The array is now 16 entries and the chunk stage count is 11.

Measured on the 13K prompt, v2, TP2, full copy, 96 GB huge pages, `DS4_COPY_PIN=1`,
`DS4_STORE_GB=0`, peer AR on:

| Config | Prefill |
|---|---:|
| before both fixes | 102 tok/s |
| direct pinned copies | 433 tok/s |
| + `DS4_ATTN_SLAB=2048` | 449 tok/s |
| + `DS4_SKIP_UNPICKED=1` | **451 tok/s** (3.5K: 331) |

That is a **4.4x** improvement, and the 3.5K prompt is no longer using the staging path
either.  Remaining gap to Maya's ~1000 tok/s is in the newer CUDA prefill kernels (attention
MMA/MMQ paths in Maya v1.1.5x) rather than in the CPU-lane copy; those are a separate port.
`DS4_ATTN_SLAB=2048` is now in the server config; `DS4_SKIP_UNPICKED=1` stays opt-in (its
layer-routing wait helps only when a chunk picks few experts; the virtual-slot base was also
wrong -- `n_slots` instead of `pv_n` -- and is fixed, but output parity is still unverified).
The safe (`DS4_ATTN_SLAB=2048`, no skip) path measured 449 tok/s at 13K; at 3.5K it measured
268 tok/s (the skip run 331, quality pending), so the short-prompt slab needs its own A/B.

## 2026-11-08: post-reboot baseline and FAT TP2 progress

After a clean reboot Maya v1 reproduces the historical figure: 13K prefill **1019 tok/s**,
6.6K **835**, decode **46.6**.  Strata on the same prompt: **551 tok/s**.  The gap is code,
not host state.

FAT_ROWS now reaches the FP16 decode of TP2's own VRAM slots (`wbase == vram_base`,
member layout), but `f16_rows` still reports an illegal access there; the prompt's
`row_tok` indices themselves are in range (verified: T 6529, row_tok 0..6528).  FAT is
kept off by default.  The remaining path to Maya's 1019 is FAT + TP_SHARES/recent slots,
not more attention.

## Launcher

`start_ds4_v2_2080ti.sh` is the 2x2080Ti launcher. It defaults to the measured configuration
(TP2, `--experts 4`, 8192 chunks, Engram 2048, CPU IQ1_S own rows, peer AR on where available) and
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

## FAT fixed, Maya's AR2, and peer AR in FP16 (2026-11-09)

- `fat_ready` sized `fat_w` / `fat_gu` for the first busy layer's FFN slice; the TP plan gives a
  device different slices on different layers (1024 and 1280 of 2304 here), so a later, wider slice
  ran past the buffers: the illegal access.  Sized for the model's `n_ff` (the widest member slice),
  as Maya does.  `DS4_FAT_ROWS=64` then ran the 13K prompt: **686 tok/s**, against 551 with the
  tiles (clean state, first prompt after load).  Still opt-in until its output is checked.
- Ported Maya's `ar2_kernel` (`DS4_AR2=1` exact F32, `2` FP16 wire), opt-in.  On this host its
  host-mapped rows and per-block progress words are slower than the peer transport:
  `DS4_AR2=2` 424 tok/s, peer AR 551-686 tok/s in the same session.
- `DS4_PEER_AR_F16=1`: the peer all-reduce's rows rounded to FP16 on the wire (half the bytes),
  opt-in.  Measured 554 tok/s against 551 the same session (the earlier 686 was a quieter host);
  inconclusive so far, so defaults are unchanged.
- Maya v1 itself, through its tuned server and config on this host, measured **540.6 tok/s** at
  13,165 tokens (24.35 s) and decode 31.2 tok/s (`clean-v1`); its engine log shows the experts'
  host time (23.4 s) as the prompt's critical path.  The historical 938-1019 runs on the same pack
  show the same 93 GB RAM copy and AR2, so the remaining difference is host state, not a missing
  algorithm.  Strata with FAT (686) is faster than Maya's own run here.

## Current-host A/B (2026-11-09, after many load/free cycles)

The same machine, one engine at a time, the same 13K prompt IDs for Strata and Maya's `clean-v1` text
prompts for Maya:

| | prefill at ~13K | decode |
|---|---|---|
| Maya v1 (tuned server, its own config) | 540.6 tok/s (24.35 s, 13,165 tok) | 31.2 tok/s (256 tok); 25.8 after an 8K warmup |
| Strata, FAT 64 + peer AR, same host state | 551 tok/s | 16.9 after the same 8K warmup |
| Strata, first run after a clean reboot | **686 tok/s** (19.07 s) | 23.7 (13K prompt) |

Maya's own engine log puts its prompt's critical path in the experts' host time (23.4 s): the same
93 GB RAM copy, the same ring, AR2 on both.  The 938-1019 runs in its log are the same algorithm and
pack on a quieter host, not a different path.  The huge-page script documents the state: repeated
load/free cycles fragment the host until a reboot; GPU clocks are 1.5-1.9 GHz at 55-67 C, so this is
not GPU throttling.

The objective's 1000 / 40+ therefore needs a **clean reboot** (and possibly the 250 W power limit)
before the next measurement.  Only `strata-ds4-hugepages` and its release script are NOPASSWD here;
`reboot` and `nvidia-smi -pl` need a password.  After the reboot:

```sh
cd /home/likan/strata-latest
sudo -n /usr/local/sbin/strata-ds4-hugepages 96
IDS=$(cat bench/ds4-baseline/prompt-13k.txt)
DS4_HIP_PEER_AR=1 DS4_ENGRAM_CACHE_MB=2048 DS4_SETTLE_MS=2000 DS4_IQ1S_ROWS=1 DS4_IQ_OWN=1 \
DS4_STORE_GB=0 DS4_COPY_HUGETLB=1 DS4_COPY_PIN=1 DS4_ATTN_V7=1 \
./build-dsv41/strata-dsv41 --pack /mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-XS-v2/maya-ds4-v2 \
  --gpus 0,1 --tp 2 --max-context 32768 --reserve-mib 768 --cpu-threads 20 --mmap-experts off \
  --experts 4 --prefill-chunk 8192 --dspark on --ids "$IDS" --n 1
sudo -n /usr/local/sbin/strata-ds4-hugepages-release
```

## After the user's reboot (250 W), 2026-11-09 09:10

The first run after the reboot (huge-page pool reserved once, no previous load):

```text
13K prefill: 18.37 s = 710.8 tok/s, FAT 64 + peer AR + V7
decode (16 tokens, 13K prompt): 29.7 tok/s; CPU lane 447 of 2670 picks
  layers 13.26 ms/token, head wait 16.89, Engram 2.93
```

That is the best measured prefill so far: **710.8 tok/s**, against 551 before FAT and 540.6 for
Maya's own tuned server on this host.  The load copied 90.4 GB in 58 s.

Every later load, even with the huge-page pool **kept reserved** and reused (two runs back to back,
`HugePages_Free=49152` between them), measured 551-561 tok/s and a 69 s copy: the pool's pages after
the first release are slower than the ones the boot reserved.  A fresh boot restores 710; merely
keeping the pool does not.  The lesson: after a reboot reserve the pool once and do not release it
between DeepSeek runs (release it only when another model needs the RAM).

Ported Maya's page-cache drop (`STRATA_DS4_LOAD_DROP_CACHE`, on): after an expert is copied into
RAM, its source pages are evicted (`posix_fadvise DONTNEED`, 90.0 GB here) instead of holding the
same bytes twice.  It removes the load's own cache pressure; it does not change the huge-page
placement above.

`DS4_ATTN_V7_DECODE=1` (opt-in): V7 for a verify window too.  19.0 against 17.7 tok/s in the same
state, and the approximate V7 arithmetic, so off by default.

Decode remains the open gap: 29.7 tok/s on the 13K prompt and 13-17 on short natural prompts,
against Maya's 31.2 on its bench and 25.8 after a warmup.  40+ is not reached.

## The first two engine starts after a boot are the fast ones (2026-11-09 10:05)

A second clean boot, the 13K prompt, one engine process a run, the huge-page pool reserved once
and **never released** between runs:

```text
start 1: load 53.8 s, prefill 18.40 s = 709.6 tok/s, decode 29.4
start 2: load 53.6 s, prefill 18.27 s = 714.9 tok/s, decode 28.0
start 3: load 64.3 s, prefill 23.21 s = 562.6 tok/s, decode 18.5
start 4: load 66.1 s, prefill 23.61 s = 553.1 tok/s, decode 17.2
start 5: 564.4, start 6: 564.0
```

So keeping the pool does **not** keep the fast state.  The third start and later are ~21% slower,
and the load's copy falls from 1.68 to 1.4 GB/s with them.  Checked and ruled out:

- huge pages vs the ordinary THP arena (`DS4_COPY_HUGETLB=0` after a release): 564.1 / 563.8,
  the same slow state;
- the learned expert profile (`--expert-profile /tmp/fixed.bin --profile-readonly`): 549.8 / 553.1;
- the model files' page cache (a userspace `posix_fadvise(DONTNEED)` over every GGUF, 1.4 GB cached
  left): 561.2, load 70.1 s;
- GPU throttling: 63 C max, 1905 MHz, 257 W; CPU package 48 C, 3.2-3.5 GHz; NVMe 46 C, no alarm;
- memory pressure: `/proc/pressure/memory` ~0, no direct reclaim.

Only a reboot restores 710.  The degradation is not tied to huge pages, the profile, or the page
cache; a fresh boot's first two 90 GB arena allocations and their DMA/CPU-lane reads are fast, later
ones are not.  A server kept running after a fresh boot keeps its arena and should stay on the fast
side; restarting the engine is what loses it.

`stop_cpu_lane.sh` now keeps the pool by default (use `--release` when another model needs the RAM).
A script that guarantees 710 per one-shot run would have to reboot first; `reboot` is not in the
NOPASSWD set here.

## Prompt lookup (Maya's --suffix-draft) wired in (2026-11-09)

Strata's `src/spec` library (suffix drafter + draft policy) existed but `ds4_main.cpp` never used it: Maya's
`Lookup` (the tokens after an earlier repeat of the context) was missing from both the CLI and the server.
Ported the glue:

- `--suffix-draft N` (default 3, `0` off), one `Lookup` per run, reset from `m.tokens()` at each request;
  the policy learns window costs and lookup acceptance as it decodes.
- A CLI natural 5347-token prompt, same host state: **15 rounds, 53 offered, 48 accepted, 4.27 tokens/round,
  27.2 tok/s** without the lookup; **12 rounds, 58 offered, 51 accepted, 5.33 tokens/round, 29.1 tok/s** with
  it (7 lookup rounds, 42 of 49 lookup drafts kept).
- Server multi-turn, round 4 (repetitive answer): **50/50 drafts, per-position up to 8 rows (past DSpark's
  block of 5), 47.4 tok/s**.  Every lookup draft is verified like a DSpark draft, so the output is unchanged.

The comparison that started this: on the same 26-token prompt and host state, Maya measured 18.34 tok/s with
103 drafts/85 kept; Strata (before the lookup) measured 17.60 with 128 drafts/86 kept.  The apparent Maya lead
was host state and prompt content, not the drafter.  The lookup is a real missing feature and is now in.

## Prompt-lookup tuning: default off until the cold start is fixed (2026-11-09)

The first wiring chose lookup windows from the draft policy's optimistic priors before any acceptance was
measured.  Two gates from Maya and a cap followed: the lookup must start with DSpark's first draft, its window
must be longer than the MTP cut, and it is capped at 8 drafts; the policy margin is 0.25.

Measured on an 832-token JSON-list prompt, 96 greedy tokens, one host state:

```text
--suffix-draft 0 (off): 31.3 tok/s, 26 rounds, 3.69 tokens/round
--suffix-draft 3 (first wiring): 20.4 tok/s, 103 drafts offered, 73 kept
+ gates: 23.8 then 26.9 tok/s (the cold-start probe still verifies an 8-row window for 1 kept draft)
```

On long repeats it still pays: a 4-turn conversation's last answer verified 50/50 drafts, per-position rows
1..8 all kept, **47.4 tok/s**; the 5347-token natural prompt went 27.2 -> 29.1 tok/s with 7 lookup rounds.
So `--suffix-draft` defaults to **0 (off)**; `--suffix-draft 3` turns it on for code/quoted/list content, and
the next step is to make the cold start safe (lookup only when the MTP cut is 0, or a conservative prior until
the first acceptance) so it can be on by default.

## Prompt lookup: cold-start gate, default back on (2026-11-09)

The regression was the policy's cold start: its optimistic priors chose 8-15-row lookup windows before it had
seen one accepted draft.  The gate now:

- before the first success, only a **match of 24 tokens or longer** probes a lookup window beside DSpark;
- after a lookup round, it must have kept **at least half** of its drafts to keep using windows;
- a lookup only runs where it starts with DSpark's first draft and is **longer** than the MTP cut, capped at 8;
- when no DSpark draft passes the floor, the lookup may carry the round alone (a bootstrap window of at most
  4 drafts), so a failed probe costs only its own rows.

Measured, same host state, 832-token JSON list (96 greedy tokens): `--suffix-draft 0` 29.7 tok/s;
`--suffix-draft 3` 27.5 tok/s with **no lookup rounds** (the match is under 24 and nothing was proven) - the
2 tok/s is the host's drift, the rounds/drafts are the same.  The 5347-token natural prompt with the gate:
**13 rounds, 50/50 drafts, 4.92 tokens/round, one lookup round 8/8 kept, 29.9 tok/s**.  The earlier ungated
JSON run was 20.4-26.9 against 31.3 without lookup; that is gone.

`--suffix-draft` is back to **3** (Maya's default) now that the cold start is gated.  The multi-turn repetitive
answer still reached 50/50 drafts and 47.4 tok/s.

Gated and default on, the same 4-turn conversation through the server: **24.0 / 25.7 / 38.9 / 44.1 tok/s**,
the fourth answer verifying 50 of 57 drafts with positions 1..7 kept (the lookup's rows beyond DSpark's 5).
So the cold-start gate keeps the long-repeat win while the JSON list runs no lookup.

## Attention slab 2048: q_b 5.10 -> 3.49 s on the 13K prompt (2026-11-09)

The chunked prompt's q_b + rope stage was decoding wq_b's EXL3 rows to FP16 once per attention slab; the
automatic slab is 8 MB (512 rows here, CUDA0 has only ~228 MiB spare), so the same weight was reconstructed
about 26 times a chunk.  Measured on the same 13K prompt, one host state per pair:

```text
default (slab 8 MB, W16 16 MB): q_b + rope 5.10 s
DS4_ATTN_SLAB=1024:             q_b + rope 3.76 s
DS4_ATTN_SLAB=2048:             q_b + rope 3.49 s   (23 fewer experts a card, 12.73 vs 12.81 GB)
DS4_W16_MB=64 alone:            q_b + rope 4.93 s   (no gain: the slab, not the scratch)
```

`DS4_ATTN_SLAB=2048` is now in both `tools/strata-ds4-v2-2080ti*.json`.  At the same host state the default
run was 662 tok/s and the slab-1024 run 719 tok/s; the slab change is worth roughly 0.5-0.8 s of the 13K
prompt, the rest of that pair's difference was the host.  Maya's own q stage (826 ms) still shows a ~4x gap in
this stage that the slab does not close.

## Lazy drafter: not applicable on TP2; lookup chain wired (2026-11-09)

Maya's lazy drafter (`draft_lazy`) only turns on when the **head group has one member**:
`lazy = draft_lazy && hg.size() == 1 && dump_d == nullptr && (conf_min > 0 || cum_min > 0)`.  A TP2 run's
head group is both GPUs, so Maya's own condition keeps the lazy path off there; there is nothing to port for
this host.  (The measured draft cost is 6.2-6.5 ms a round, so the ceiling was small anyway.)

Lookup chain (`--lookup-chain N`, default 0 off): after DSpark's kept prefix, `SuffixDrafter::propose_after`
extends the lookup and `DraftPolicy::chain` decides how many rows to append, up to 8.  Natural 5347-token
prompt, same host state, 64 tokens:

```text
--lookup-chain 0: 13 rounds, 50/50 drafts, 4.92 tokens/round, 16.69 tok/s
--lookup-chain 4: 13 rounds, 55/50 drafts, 4.92 tokens/round, 16.50 tok/s
                 2 chain rounds, 3 of 7 chained drafts kept
```

Neutral on that prompt (the chain rows were mostly rejected); it stays opt-in for long repeats.  Every chained
draft is verified like any other, so the output is unchanged.

The chain's multi-turn check (server, `--lookup-chain 4`, same 4 turns): 22.1 / 29.3 / 28.4 / 35.9 tok/s,
against 24.0 / 25.7 / 38.9 / 44.1 without it - the chain's rows were rejected and cost more than they added.
The fat config keeps the option off; it stays available for long, exactly repeated text.

## Flow TP in the slow host state (2026-11-09)

13K prompt, slab 2048, same period:

```text
DS4_FLOW_TP=0: 23.55 s (554.5 tok/s), experts 12.28 s, ffn mHC+router+sort 9.18 s, between layers 0.46 s
DS4_FLOW_TP=1: 21.16 s (617.1 tok/s), experts 16.26 s, ffn mHC+router+sort 2.88 s, between layers 0.03 s
```

The flow trades a larger expert stage for a much smaller host-side ffn/between stall; the first token is 455 in
both.  It is **not** enabled by default: in the clean fast state no-flow's ffn is already 2.8 s, so the flow's
+4 s expert stage would likely dominate.  Needs a fast-state A/B after a reboot; `DS4_FLOW_TP=1` is the switch.

## DS4_AR2_DECODE: decode all-reduces in one launch (2026-11-09)

Ported Maya's `DS4_AR2_DECODE=1`: the decode windows' all-reduces (G.ar, a group of two) run in ar2_kernel's
exact mode (1: member 0's partial plus member 1's, ar_sum_kernel's bits) instead of xfer_put + ar_sum_kernel.
Its own progress words; the chunk path (G.arb) keeps whatever `DS4_AR2` chose.  13K prompt, 16 tokens, same
slow host state:

```text
off: decode 17.62 tok/s, layers 21.50 ms/token
on:  decode 18.18 tok/s, layers 20.23 ms/token   (first token 455 in both; exact sums)
```

`DS4_AR2_DECODE=1` is now in both tuned configs.  It is exact, so no output check beyond the token match is needed.

## 220 W fast state: 731.7 tok/s prefill, 30.8 decode (2026-11-09)

The host returned to the fast state at the user's 220 W setting (no reboot).  13K prompt, slab 2048,
AR2_DECODE, peer AR, V7:

```text
ZC=1: prefill 17.85 s = 731.7 tok/s, decode 29.31 tok/s
ZC=0: prefill 17.92 s = 728.7 tok/s, decode 28.24 tok/s
fixed expert profile, ZC=0: 725.4 / 30.24; ZC=1: 724.5 / 30.79
```

- **731.7 tok/s** is the best 13K prefill measured, at 220 W (above the earlier 719 at 250 W).
- `DS4_STREAM_ZC=1` is now in both configs: prefill neutral, decode +1.8% in the fixed-profile A/B.
- Flow TP in the same fast state: **695.8 tok/s** against 731.7 without it - the flow's larger expert stage
  loses once the host-side ffn stall is gone.  It stays off.
- The v1 pack (IQ1_S down) in the same state: 561.9 tok/s and decode 11.9, its profile placing only 5,344
  experts in VRAM and 41% of decode picks on the CPU lane.  The v2 pack stays.

## 220 W multi-turn with ZC + AR2_DECODE + slab 2048 (2026-11-09)

The same 4-turn conversation through the server: 23.0 / 25.3 / 33.8 / 33.6 tok/s, CPU lane 36.0 -> 25.2%.
The earlier 44.1 / 47.4 rounds were a particular answer's repeating pattern with lookup windows to 7-8 rows;
the slab change moves the prompt GEMM's rounding, so the greedy answer (and its repetition) differs and the
lookup window no longer extends.  The 13K decode is the stable number: 30.8 tok/s; 40+ appears on warm
conversations whose answers repeat, and is content-dependent, not a floor.

## Copy-pipeline experiments at 220 W (2026-11-09)

The remaining prefill gap is the expert copy stream: a layer's ~2.25 GB into a 400 MB ring, 6.6 s of expert
GEMM and ~2.9 s of waits over the 13K prompt.  Tried and measured:

```text
early copy_ahead at the start of each layer (before attention, chunks that copy all): 727.2 tok/s,
  experts 9.59 s - no gain over 728-732; reverted (the first batches were already covered)
DS4_RING_MB=800, fixed profile: prefill 726.4 / decode 29.8; ring 400 in the same pair 645.3 / 30.1
  (the pair's first run caught a slow host state; no reliable ring gain)
Chunk::kParts 8 (finer batches, same 400 MB): 719.3 tok/s, experts 10.05 s - worse; reverted
DS4_FLOW_TP=1 in the fast state: 695.8 tok/s - worse; stays off
```

The copies aggregate ~9.5 GB/s over the two PCIe 3.0 links (each 12.2 GB/s), with the 2.9 s of waits spread
over the layers; the ring depth and batch granularity are not the limit in these tests.  Next would be a
persistent per-expert copy schedule or more VRAM experts, not another ring size.

## The v1 pack was slow because of its profile, not its quant (2026-11-09)

Earlier the v1 pack (IQ1_S down, larger dense weights) measured 561.9 tok/s with 41% of decode picks on the
CPU lane.  The same pack with the v2 pack's learned expert profile (`--expert-profile <v2>/expert-profile.bin
--profile-readonly`) at the same fast state:

```text
v1 pack + v2 profile: prefill 18.38 s = 710.6 tok/s, decode 32.71 tok/s, CPU lane 504 of 2670 (18.9%)
v2 pack (default):    prefill 17.85 s = 731.7 tok/s, decode 30.8 tok/s
```

The profile was the whole difference (CPU lane 41% -> 18.9%).  v1 trades 3% prefill for 6% decode: its down
matrices are IQ1_S (1.06 bpw) instead of IQ2_XXS, but its dense weights are 0.9 GB larger, so fewer experts fit
VRAM.  The default stays v2 (higher quality, faster prefill); v1 is an option for decode-heavy use.

## Hybrid pack: v2 dense + v1 IQ1_S down (2026-11-09)

Built with `tools/make_hybrid_pack.py`: the v2 pack's dense weights, gate and up (smaller than v1's dense),
with the v1 pack's `w2` (down) experts, IQ1_S instead of v2's IQ2_XXS.  15,360 w2 tensors, 10-12 files,
the GGUFs symlinked.  Config: `tools/strata-ds4-hybrid-2080ti.json` (v2 learned profile, read-only).

13K prompt, same fast state, 220 W:

```text
hybrid: prefill 17.75 s = 735.6 tok/s, decode 31.90 tok/s
        CPU lane 87.1 GB copied, 5,900 VRAM experts (v2: 90.4 GB, 5,746)
v2:     731.7 / 30.8
v1:     710.6 / 32.7
```

4-turn server conversation: 23.1 / 24.6 / 25.5 / **47.3 tok/s**, the last answer 50 of 51 drafts with
positions 1..5; v1's same conversation reached 53.0.  The hybrid is the best prefill and a better warm decode
than v2, at v1's lower down precision.

Copy microbenchmark (pinned H2D, 5120 rows): the down's 2D copy (width 200, pitch 450) 8.76 GB/s, a 1D 1 MB
copy 11.8 GB/s - only 26% apart, not the prompt's limit; the aggregate copy waits are the scheduling limit.

## The member's down is already sliced: no DOWN_PART port needed (2026-11-09)

`DS4_PROF` now prints what each device actually DMAs.  Hybrid pack, 13K prompt, two 8192 chunks:

```text
CUDA0: 25,204 experts, gate/up 55.50 GB, down 27.75 GB (1.10 MB down each)
CUDA1: 25,204 experts, gate/up 60.64 GB, down 30.32 GB (1.20 MB down each)
```

The full IQ1_S down row is 2304/256*50 = 450 B; a full 5120-row down is 2.304 MB.  The member's 1.10-1.20 MB
is 44-56% of that: `stream_expert`'s 2D copy uses width `dh = type_row_bytes(t2, x.ffn)` and source pitch
`dr = type_row_bytes(t2, g.ff)`, so it already transfers only the member's columns.  Maya's
`DS4_FLOW_DOWN_PART` compacts the **flow** path's staging because that path DMA'd whole down rows; the CPU-lane
copy path never had those bytes to save.

`DS4_FLOW_DOWN_SHARE` is the other half: two members reading the same entire down range from the files.  Our
`--mmap-experts off` copies each expert's file range once at load (87.1 GB) and both members stream their
halves from that one RAM copy, so there is no second file read to share either.  The flow path is off in the
fast state (695.8 against 731.7 tok/s), so neither flag applies.

The 13K stream is 174 GB over two chunks: 116 GB gate/up and 58 GB down.  The down slice is already minimal;
the next lever is the gate/up bytes (a lower-bit pack) or more VRAM experts, not DOWN_PART.

## Gate/up inventory and the ring ceiling (2026-11-09)

Every available pack's routed gate/up is **IQ1_S** (30,720 tensors; the drafter's 768 are IQ2_XXS):

```text
Maya-XS/maya-ds4 (v1):       gate/up IQ1_S 30,720, IQ2_XXS 768; down IQ1_S 15,360, IQ2_XXS 384
Maya-XS-v2/maya-ds4-v2:      gate/up IQ1_S 30,720, IQ2_XXS 768; down IQ1_S 10,752, IQ2_XXS 4,992
Maya-XS-hybrid:              gate/up IQ1_S 30,720, IQ2_XXS 768; down IQ1_S 15,360, IQ2_XXS 384
Maya-S:                      only 4 of 6 shards present, IQ2_XXS (higher bits than IQ1_S)
```

IQ1_S (50 bytes / 256 values = 1.5625 bpw) is already the lowest type the engine has kernels for; a lower gate/up
needs a new quant type and the pre-quant weights, which are not in the packs.  The 116 GB of the 13K gate/up
stream is at that floor.

A whole-layer ring was the last scheduling idea: `DS4_RING_MB=2500`, hybrid 13K:

```text
prefill 17.98 s = 726.4 tok/s, experts 9.17 s (ring 400: 17.75 s, experts ~9.5)
VRAM experts 2,727 + 2,343 = 5,070 (ring 400: 5,900), stream 26,034 experts a device
```

The deeper ring cut the expert stage by ~0.35 s but lost 830 experts, adding 1,660 expert streams and ~3.3 GB of
gate/up a device; the total was slightly worse.  Combined with the earlier ring 800 / kParts 8 / early copy_ahead
/ flow tests, the copy pipeline is at its scheduling limit for this quant and 44 GB of VRAM.

## Persistent per-expert copy pump (DS4_COPY_PUMP=1) (2026-11-09)

A thread of its own issues the streamed experts' batches as ring slots free; the main thread only waits for
the batch's filled event (`cp > kp` then `cudaStreamWaitEvent`).  It is gated to the full-copy chunks
(`skip_unpicked_rows` 0), the copied-and-pinned RAM source, and the non-flow path; the streamer's `cp`/`kp` are
atomics and the pump reports a copy error through `S.bad`.  Hybrid, 13K, same state:

```text
pump:    17.69 s = 738.0 tok/s, experts 9.20 s
no pump: 17.75 s = 735.6 tok/s, experts 9.24 s
first tokens identical: 455,2502,344,260,1606,...
```

The persistent schedule removes the host gaps between `copy_ahead` calls (about 0.3% here); the ring slots
and the copy/compute balance still set the pace, so it is a small but exact gain.  Enabled in the hybrid and
v2 tuned configs.

## DS4_ROUTE_FUSE: the gate GEMV pair ported (2026-11-09)

Ported Maya's `gemv_f32_pair_kernel` / `gemv_f32_pair_rows_kernel` (F32): the decode layer's router gate and the
next layer's lookahead gate (the guess the prefetch already made) in one launch.  Each row is gemv_f32_kernel's
code, so the bits match; `DS4_ROUTE_FUSE=0` turns it off (default on).  Hybrid 13K, same state, 16 tokens:

```text
off: decode 26.29 tok/s, layers 10.65 ms/token
on:  decode 26.50 tok/s, layers 10.57 ms/token; tokens identical
```

The routing/guess kernels (`route_pair`) are not ported yet: Strata's route_publish carries the plan pointer,
the VL bits and the cache-substitution argument, and the fused kernel would have to reproduce that state.  The
gate pair alone is a small exact gain (~0.8%).

## DS4_ROUTE_FUSE: route_pair ported (2026-11-09)

The second half: `route_pair_kernel` (block 0 route_publish_kernel's selection, block 1 route_ahead_kernel's
picks, row blockIdx.y) plus the `RoutePair` struct and the `publish_kernel` follow-up.  It is used only when
the cache-substitution router is off (the fused block has the exact publish body); otherwise the two launches
stay.  Hybrid 13K, pump + gate pair + route pair, same state:

```text
prefill 17.64 s = 740.1 tok/s   (best measured)
decode 28.00 tok/s, layers 10.65 ms/token
tokens identical: 455,2502,344,260,1606,...
```

The route pair removes one more launch a layer; the decode's per-token time is set by the CPU lane/expert
tail, so the gain is within run-to-run noise.  `DS4_ROUTE_FUSE=1` (default) enables the gate pair and the route
pair; `0` disables both.

Gated and default on, the same 4-turn conversation through the server: 23.3 / 24.9 / 25.6 / **48.0 tok/s**
(the fourth answer 50 of 51 drafts, positions 1..5) - the best warm decode measured, against 47.3 before the
route pair and 53.0 for the v1 pack.

## Clean reboot validation: 759.6 tok/s (2026-11-09, 250 W)

Fresh boot, the huge-page pool reserved once and kept (not released), hybrid + copy pump + route fuse:

```text
13K prefill: 17.19 s = 759.6 tok/s   (new best; 740.1 before the reboot, 2.6% higher)
13K decode:  26.63 tok/s in this boot/profile
4-turn:      23.3 / 21.9 / 30.0 / 36.6 tok/s
```

The 4-turn answers this boot were a different text (fresh-boot allocation order moves the GEMM rounding, so
the greedy answer differs); the earlier 48.0 and 53.0 warm peaks were other answer shapes.  Prefill is the
stable metric: the clean boot adds 2.6%, and the remaining gap to 1000 (24%) is the IQ1_S gate/up stream and
the 44 GB of VRAM, not host state.  The pool is kept reserved for the next start; stop with `--release` only
when another model needs the RAM.

## Same-boot A/B: Maya 1018 against Strata 759.6 (2026-11-09)

Fresh boot, the huge-page pool reserved once and kept:

```text
Strata (hybrid + pump + route fuse): 13K 17.19 s = 759.6 tok/s
Maya v1 (same pool, then started):    warmup 10685 tokens 1000.2 tok/s
                                      clean-v1 13162 tokens 12.93 s = 1018.1 tok/s; decode 45.5
```

So the gap is real code/configuration, not host state.  Checked:

- cache substitution (Maya's 0.1): 580.5 against 579.1 without it - prefill neutral (it changes the decode's
  CPU lane, not the prompt's expert set).
- q_b's GEMM: `DS4_GEMM_PROF` shows reconstruct 0.16 ms, cuBLAS 1.4-1.8 ms a slab, ~80 ms a device a prompt.
  The 3.45 s "q_b + rope" stage is the profiler's marks inside the slab loop, not the GEMM.
- host time: `host_layer_ms` 15.5 s, of it `sync_ms` 14.3 s - the main thread is only ~1.2 s of CPU; the
  prompt is GPU-stream-bound, not host-enqueue-bound.
- ring 2500 + pump: host 15.6 s, 733.8 tok/s - the deeper ring did not move the sync.

The remaining 4.3 s is in the GPU stream's per-layer work (attention ~7.5 s a device, expert copy waits,
2.2 s of all-reduces); the profiler's per-stage sum is not a trustworthy attribution there.

Peer AR in FP16 (`DS4_PEER_AR_F16=1`), hybrid 13K, same slow state: prefill 746.9 tok/s (17.48 s) against
736.6 (17.73 s) for the F32 peer AR, host sync 14.07 against 14.33 s - about 1.4%, and the wire is
approximate (both partials rounded to FP16).  It stays opt-in; the default remains F32.

## The 1018 tok/s benchmark is a warm, repeated-text read (2026-11-09)

Maya's `bench_ds4.py` prompt is not a document: it is one ~60-character paragraph repeated to the requested
length (`BASE * int(n * 1.4 / len(BASE))`).  Its 13K run at 1018 tok/s is the **second** read of that prompt in
the same session, after a 2-round warm-up.  The engine log from that run:

```text
maya ds4 serve: prompt 13165 tokens = 0 reused + 13165 read in 12919 ms (1019.0 tok/s)
  CPU lane 0.0% of the experts ... expert cache 98.9% hit
```

Cold, the same engine and the same 13,058-token prompt took **96.5 s (135 tok/s)** on the timing pick's windows
path and **125.4 s (104 tok/s)** forced to the chunked path; Strata's cold read of the same prompt is 17.8 s
(734 tok/s).  Maya's own docs (`docs/2080ti22gx2/DS4_V2_EXPERT_CACHE.md`, `BENCHLOG.md`) say the ~1000 number
needs `DS4_SETTLE_MS=2000` + two warm-up rounds and that repeated model reloads leave the host below that until
a reboot.

Same boot, same repeated prompt, same 2-round warm-up (220 W, pool kept):

```text
Strata hybrid + pump + route fuse: 13,162 in 17.81 s = 739.0 tok/s; decode 10.5
Maya v1, no profiler:              13,162 in 24.31 s = 541.3 tok/s; decode 31.4
Maya v1, DS4_PROF=1:               13,162 in 23.81 s = 552.9 tok/s
```

So in this degraded host state Strata's prefill is 37% faster than Maya's on the identical prompt; the
remaining decode gap is the CPU lane (Strata's per-token CLI wait 43.9 ms against 18-20 ms after a clean boot)
and needs a clean boot to separate host state from code.  The clean 1018 is a best-case warm number, not a
floor.

## DS4_TP_SHARES ported (opt-in, default unchanged) (2026-11-09)

Maya's static FFN shares, `DS4_TP_SHARES=4,5` (member m takes w_m of sum(w) whole quant blocks), added to the
GGUF split.  With `--gpus 0,1` member 0 carries the output head, so `4,5` gives the head device 1024 of the
2304 FFN and the other 1280.  Hybrid 13K, 220 W, same boot:

```text
DS4_TP_BALANCE=0 (rotation, 4/5 round the layers): 738.5 tok/s, streams 58.65 / 58.68 GB
default balance planner:                           734.0 tok/s, streams 55.50 / 60.64 GB
DS4_TP_BALANCE=0 DS4_TP_SHARES=4,5:                734.6 tok/s, streams 50.78 / 63.48 GB
```

The head-side 1024 is not a win here; the balanced rotation is best and the default stays.  The share env is
kept for packs/layouts where the output head's device needs the smaller slice.

## Warm-up added (2026-11-09)

`tools/warmup_ds4.py` (Maya's script) sends a short decode, 1.6K and 13K reads and a decode, twice; placement
only, no output change.  Run it after `/health`:

```sh
./.venv/bin/python tools/warmup_ds4.py --rounds 2
```

`STRATA_WARMUP=1 ./start_cpu_lane.sh` does the same after `/health` (rounds from `STRATA_WARMUP_ROUNDS`, 2 by
default); the default stays off so a start returns at once.

Without it the first real request pays the expert-cache fill: in this state the first short decode measured
5.9 tok/s and the first 1.6K read 117 tok/s, then 28.0 and 545 tok/s after the two rounds.

## Clean boot, 220 W: 810 tok/s on the repeated prompt (2026-11-09)

Fresh boot, the huge-page pool reserved once, `STRATA_WARMUP=1` (two rounds), hybrid + pump + route fuse:

```text
warm-up: 1.3K 199.3 s/6.57 then 251.9/5.20; 10.7K 700.6 s/15.25 then 802.6/13.31; decode 30.2
bench:   1,615  4.55 s = 355.3 tok/s
         6,560  7.94 s = 826.4 tok/s
        13,162 16.25 s = 810.1 tok/s; decode 17.4 (256-token Python prompts)
```

The earlier clean-boot Maya reference on the same benchmark (same machine, 220 W) was 507.7 / 825.4 /
**1018.1** tok/s and decode 45.5.  The 6.6K point is at parity; the 13K is 80% of Maya's and the decode is
content-dependent.

State gauge: the warm-up's second 10.7K read is **803 tok/s** on the first two loads of this boot.  After
three more loads it fell to 501/545 and the bench to 649; Maya, started after three Strata loads in the same
boot, also fell to 485 warm-up / 548 bench (its earlier clean boot: 1000/1018).  A boot therefore gives about
two fast loads; past that the CPU-lane arena degrades and both engines lose ~35-45% until the next reboot.
The warm-up rate is the cheapest gauge.

Serve against batch, same 512-token prompt and 256 generated tokens: serve **37.75 tok/s** (6,780.9 ms) against
batch 37.0 (6,892 ms), so the 17.4 on the Python prompts is not a serve-path cost.  In one serve session the
same 26-token Python prompt decoded **10.7 tok/s cold and 37.4 tok/s warm** (fourth read), with the draft
positions 1-5 at 83-100%: the 40+ decode needs the expert cache warm on that content, exactly Maya's
`vram_hit 0.93-0.99` condition.

## Short-prompt transfers: skip the experts a chunk does not pick (2026-11-09)

Maya's CUDA default is `DS4_SKIP_UNPICKED_ROWS=1536` (a chunk up to that many rows copies only the experts some
row picked); Strata defaulted to 0 (every chunk copies every CPU-lane expert).  Set to **2048** so a 1.6K
agent turn skips, while the 6.6K/13K chunks keep the full-copy path that overlaps better:

```text
same degraded state (warm-up 10.7K 543 tok/s both), bench:
  1,615 tokens: 9.21 s = 175.0 tok/s   without
                4.08 s = 396.1 tok/s   with 2048   (2.3x)
  6,564 tokens: 652.6 against 654.5    (unchanged)
 13,160 tokens: 645.5 against 649.8    (unchanged)
warm-up 1.3K:  116.6/117.8 tok/s      without
                208.4/222.8 tok/s      with
```

Placement/order of the copied experts is unchanged, so the outputs are the same; the copy of an expert no row
of the chunk picked is simply not issued.  Written into the hybrid and v2 tuned configs.

## Cross-layer copy overlap: kParts 8 + ring 800 MB measured, no gain (2026-11-09)

The prompt path's four ring parts hold one layer's batches; the next layer's copies wait for them.  Eight parts
with an 800 MB ring should let the next layer's copies run during the current layer's attention.  Same degraded
state (warm-up 10.7K 540 against 543), hybrid bench:

```text
kParts 8 + ring 800: 1,613 3.89 s = 415.2; 6,564 10.07 s = 652.1; 13,161 20.44 s = 643.9
kParts 4 + ring 400: 1,615 4.08 s = 396.1; 6,564 10.06 s = 652.6; 13,160 20.39 s = 645.5
```

No change on the long chunks (the 1.6K difference is the skip path's noise), so the ring depth and part count
are not the remaining 13K cost; reverted to 4/400.
