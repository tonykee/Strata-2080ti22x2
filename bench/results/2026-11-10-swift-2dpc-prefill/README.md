# Swift 1.5 IQ3_S: 256 GB 2DPC against the 128 GB 1DPC record (2026-11-10)

Status: **local record, one run per cell unless said otherwise.**  This is the follow-up to the 256 GB
memory upgrade: the 128 GB (1DPC) box reached 128.2 tok/s decode at 50K and 2,378.9 tok/s prefill at
100K; after moving to 8x32 GB (2DPC) the same test dropped.  This note records how much of that drop is
the memory configuration and which host-side settings recover part of it.

## Machine and engines

- 2x GeForce RTX 2080 Ti 22 GB, 220 W, Xeon E5-2666 v3 (10 cores / 20 threads), HuananZhi X99-T8.
- **Current:** 256 GB DDR3-1600 4Rx4, 8x32 GB, **2DPC**, Linux 7.0.0-42, driver 595.91.07, engine **0.1.57**
  (branch `ds4-cuda-prefill`, HEAD `2519c0ef`), config `strata-swift-latest.json` (393216 context, rope 1.5).
- **Old record:** 128 GB DDR3-1600 4Rx4, 4x32 GB, **1DPC**, 2026-11-03/04, engine **0.1.45**, 524288
  context, rope 2.0 (`/home/likan/strata/strata-swift-iq3_s-512k.log`, lines 311/313/321).
- **Control:** 0.1.45 on the current 256 GB box, same 393216/1.5 config unless noted.

## Test

The old `swift_x99_bench.py` protocol, recovered and re-run unchanged: a segment filler sized by the
tokenizer, ending with

```text
List the integers from 1 to 500, one per line, with no other text.
```

20K / 50K / 100K prompt tokens, `max_tokens=512`, temperature 0.  The engine log is the source of the
prefill and decode figures; the client wall time and GPU samples are in `data/`.  The integer-list tail is a
best case for the draft layer (95%+ acceptance); it is not a general chat decode.

## Results

### The record and the same-config control

| Test | old 0.1.45 / 128 GB 1DPC | 0.1.57 / 256 GB 2DPC | 0.1.45 / 256 GB 2DPC |
|---|---:|---:|---:|
| 20K prefill | 1,768.6 | 1,809.2 | 1,575.2 |
| 20K decode, first / warm | 96.0 / - | 109.5 / 136.7 | 57.7 / 111.9 |
| 50K prefill | 2,182.7 | 1,813.7 | 1,826.9 |
| 50K decode | 128.2 | 116.0 | 111.4 |
| 100K prefill | 2,378.9 | 1,977.9 | 1,996.8 |
| 100K decode | 125.8 | 121.0 | 117.3 |

The 20K row is the dedicated two-run test (`data/x99-20k-0157.json`, `x99-20k-0145.json`); its first run
is cold-cache (90-91% expert-cache hit) and the second is warm (97.8-98.1%).  The 50K/100K rows are one
sequence each.  Current 0.1.57 is 3-4% faster than 0.1.45 on the same 256 GB box at 50K/100K decode and
within 1% on prefill, so the drop against the 128 GB record is the memory state, not the current branch.

### Host-side settings tried (all on 0.1.57, 256 GB)

| Setting | 50K prefill | 100K prefill | 50K decode | 100K decode | Verdict |
|---|---:|---:|---:|---:|---|
| current default | 1,813.7 | 1,977.9 | 116.0 | 121.0 | baseline |
| 2 MB hugepages for the expert arena | 1,894.6 | 2,064.7 | - | 119.6 | **keep: +4.5% prefill** |
| + `--vram-reserve-mib 700`, no later reserve | 1,935.3 | 2,094.3 | - | 114.2 | **keep: +1.5-2%** |
| + `--batch 1` (batch slots off) | 1,934-1,979 | 2,111-2,133 | 93.7 first / 133.7 warm | 131.5-133.6 | **best: cache 9009+7963 slots** |
| `--prefill 4096` | 1,283.7 | 1,337.0 | 61.1 | 118.9 | no: much worse |
| `--pool-workers 12` | 1,814.5 | 1,988.0 | 61.3 | 119.3 | no: worse |
| `STRATA_EXP_MODE=8` / `PW_WITH_BATCH` off | 1,839.8 | 2,007.8 | 54.2 | 89.9 | no: worse, especially decode |
| `--pipeline-windows 0` on the best config | 1,958.2 | 2,140.0 | 81.0 | 107.5 | no: decode loses 19% |

`--batch 1` is reported by the engine as "off: it needs 2 or more slots"; that is intended here.  It frees
1.53 GB per card for the expert cache.  The 50K decode depends strongly on that cache: 97.0% hit on the
first 50K run gives 93.7 tok/s, 99.6% on a second 50K run gives 133.7.  On 2DPC every missed expert is
expensive, so a warm 50K request is above the old 128.2.

### Best combination

- 2 MB hugepages for the expert arena (46.84 GiB = 23,983 pages; a 48 GB pool is enough).
- `--vram-reserve-mib 700`, no `--vram-reserve-later-mib`.
- `--batch 1 --batch-groups 1` (batch slots off; single-request serving).
- `--pipeline-windows 2`, `--prefill auto` (8192 chunk), default 9 pool workers, current env.
- Config: `/home/likan/strata-latest/strata-swift-2mb-cache.json` (ignored by git; the flags above are the
  record).

Result: 50K prefill **1,934-1,979** and 100K **2,111-2,133** (against 1,813.7/1,977.9 baseline, +7-9%;
against 2,182.7/2,378.9 on 1DPC, -9-11%).  Decode: 100K **131.5-133.6** and warm 50K **133.7** (above the
old 125.8/128.2); first 50K 93.7 because the expert cache is still at 97.0%.

Run it:

```sh
# 48 GB of 2 MB pages is enough; the 96 GB helper also works but wastes 48 GB.
sudo sysctl -w vm.nr_hugepages=24576
# or, with the installed helper: sudo /usr/local/sbin/strata-ds4-hugepages 96

cd /home/likan/strata-latest
./bin/strata serve --profile strata-swift-2mb-cache.json
```

The engine log must say `hugetlb 2 MB pages`; if it says `MAP_HUGETLB unavailable` it is on THP/4 KB pages
and the prefill gain is absent.  Release the pool with
`sudo /usr/local/sbin/strata-ds4-hugepages-release` when the model is stopped.

## What remains

The best config still leaves about **9-11% prefill** below the 1DPC record; both engines lose the same
amount on 2DPC, so this is the memory layout, not the code.  The BIOS exposes no rank-interleaving or
command-rate setting.  Remaining options:

1. **1 GB hugepages for the arena.**  The kernel has `hugepages-1048576kB` and the CPU has `pdpe1gb`, but
   the 1 GB pool is 0 and `src/core/pinned.cu` only tries `MAP_HUGE_2MB`.  An opt-in `MAP_HUGE_1GB` path
   (47 TLB entries instead of 23,983) plus `default_hugepagesz=1G hugepagesz=1G hugepages=48` is the next
   code experiment; expected a few percent, not the full gap.
2. **PLE table hugepages.**  The 26.8 GiB PLE table is mmap + mlock (`--ple-io ram`) and uses page-cache
   pages; copying it into a 2 MB pinned arena is a code experiment, expected a few percent.
3. **1DPC for model runs.**  Four 32 GB DIMMs (128 GB) recover the full prefill record (2,183/2,379) and the
   128 tok/s decode; eight DIMMs keep 256 GB but pay the 2DPC penalty.

## Files

- `data/x99-0157.json`, `data/x99-0145.json`: the 20K/50K/100K record on 256 GB, 0.1.57 and 0.1.45.
- `data/x99-20k-0157.json`, `data/x99-20k-0145.json`: the dedicated two-run 20K test.
- `data/x99-hp-0157.json`: 2 MB hugepages.
- `data/x99-vram700.json`, `data/x99-batch1.json`, `data/x99-best.json`: the cache settings; `x99-best.json`
  holds the 50K/100K/50K sequence (first vs warm 50K).
- `data/x99-pf4096.json`, `data/x99-pw12.json`, `data/x99-hp-envoff.json`, `data/x99-batch1pw0.json`: the
  rejected settings.
- `data/x99-512k.json`: 0.1.57 with the old 524288/rope 2.0 config.
- `bench.py`: the harness used (adjust `ROOT`, `TOKDIR`, `ENGINE_LOG`; it expects a running server).
