# Qwen3.8-Flash-Next: candidates ported from the DeepSeek-V4.1 work

Status: **candidates only, not implemented for Qwen.**  Every number below was measured on the DeepSeek-V4.1
engine (`src/dsv41/`) on 2x RTX 2080 Ti 22 GB, 220 W, IQ1_S-class experts, unless it says otherwise.  The Qwen
engine is `src/qwen35/q36_engine.cu`; none of the dsv41 code is shared, so each candidate needs its own A/B on a
Qwen pack before it is believed.  This note exists so the measurements and the exact change points are not lost.

## What q36 already has (do not port these)

| Mechanism | Where |
| --- | --- |
| Start warm-up (chunk, small prompt, decode, MTP) | `Engine::warmup()`, `q36_engine.cu` ~2758; `--no-warmup` |
| Post-prompt expert settle | `after_rows()` ~1282, `rebalance()` ~2299, called after the prompt ~2487/2491 |
| Routing counts / hot set | `Hy::score`, halved at each rebalance |
| Expert placement / profile | `place_experts()` ~2800 |
| Two-layer expert prefetch | `stream_begin`/`stream_kick`/`stream_take` ~2168-2225, double-buffered events |
| Approximate route preference for cached experts | `STRATA_ROUTE_PRIOR` (release notes 0.2.7), distinct from DS4's cache substitution |

## Candidate 1 (highest expected value): short chunks stream only the picked experts

**Current q36.**  `stream_kick()` copies a whole layer's expert blobs into the staging buffer: runs of
VRAM-slot experts device-to-device, the rest host-to-device, or the whole layer from host when
`STRATA_Q36_STREAM_ALL=1` or fewer than half the experts have a slot.  It does not look at which experts the
chunk's router picked.

**Measured on DS4 (the same idea).**  `DS4_SKIP_UNPICKED_ROWS=2048` makes a chunk of up to 2048 rows wait for
its routing and copy only the experts with rows:

```text
same degraded state (warm-up 10.7K 543 tok/s both), hybrid pack, Maya bench prompt:
  1,615 tokens: 9.21 s = 175.0 tok/s   without
                4.08 s = 396.1 tok/s   with 2048        (2.3x)
  6,564 tokens: 652.6 against 654.5    (unchanged)
 13,160 tokens: 645.5 against 649.8    (unchanged)
warm-up 1.3K:  116.6/117.8 tok/s      without
                208.4/222.8 tok/s      with
CLI 1,600-token prompt: 12.00 s = 133.3 tok/s -> 5.67 s = 282.0 tok/s
  generated tokens identical: 10021,9162,943,3277,34788,25884,14,6464,...
```

The gain is on short chunks (agent turns, chat), where the routing touches a small part of the CPU-lane set;
long chunks keep the full-copy path because their routing touches nearly everything and the wait costs more
than it saves (`docs/ds4-skip-unpicked.md` has the same conclusion from Maya's side).  Commit `ce3c79b75`.

**For q36.**  Add a short-chunk path to `stream_kick`: for chunks below a row threshold, compute the router
first, read the group counts/bounds (`k_route_scan` already produces them), and copy only the experts with
rows (VRAM-slot ones still D2D).  Keep the current whole-layer/double-buffered path for long chunks.  The
stream is prefetched two layers ahead today, so this needs a synchronous short-chunk variant (router, then
copy, then experts), exactly the shape DS4 uses.

**Validation.**  A 1.6K prompt and a 13K prompt, fixed nonce, temperature 0, greedy ids byte-compared with the
current path; prefill tok/s and a 256-token decode; clean boot (see Candidate 3).  Expect the short prompt to
move, the long one to stay.

## Candidate 2: the host expert arena on 2 MB huge pages

**Current q36.**  The hybrid host arena is one `cudaHostAlloc(host)` allocation (~the CPU-lane experts' bytes,
`L.hblobs` inside it, `q36_engine.cu` ~2924-2956); it is pinned but 4 KB-page backed.

**Measured on DS4.**  The prompt copy arena uses `MAP_HUGETLB` 2 MB pages (`DS4_COPY_HUGETLB=1`, pool from
`strata-ds4-hugepages`).  A fresh boot's pool is the fast state: the 10.7K read runs at ~803 tok/s; after three
model loads the same read is ~540 and the 13K bench falls from 810 to ~645.  Releasing and re-reserving the pool
does not restore it (Maya's docs record the same "reload until reboot" behaviour).  The pool is released only
when switching models (`stop_cpu_lane.sh --release`), and Qwen does not use it today.

**For q36.**  Reserve the pool once, `mmap(MAP_HUGETLB)` the arena, `cudaHostRegister` it, and fall back to the
current `cudaHostAlloc` when the pool is absent.  The current hugepage scripts are DS4-named but generic; the
release script already says the pool is returned for Qwen or anything else.

**Validation.**  Same q36 pack and prompts on a clean boot, arena on/off (two boots if necessary; one boot
supports only about two fast loads).  Watch the 10.7K warm-up rate, prefill 1.6K/6.6K/13K and decode.

## Candidate 3 (no code): the clean-state benchmark discipline

From the DS4 run, the machine — not the engine — sets the ceiling:

```text
clean boot, first/second load:  warm-up 10.7K 803 tok/s, 13K 810 tok/s
after three more loads:         warm-up 10.7K 540-547, 13K 645, decode 11 (was 31.9)
Maya, same boot after 3 loads:  warm-up 485, 13K 548 (its clean boot: 1000-1018)
```

Use the **second 10.7K warm-up read as the state gauge**: ~800 is fast, <600 is degraded.  Run no more than
about two model loads between reboots for benchmarks; keep the huge-page pool reserved across restarts and
release it only to switch models.  Compare cold and warm runs of the *same* prompt; a warm second read of a
repeated prompt (Maya's bench) is not comparable with a cold read of a different one (this was the source of
the "Maya 1018 against Strata 759" confusion).

## Low-confidence candidate: IQ1_S CPU rows for the CPU pool

The DS4 CPU lane's IQ1_S AVX2 row kernels (`src/dsv41/ds4_iq_cpu.cpp`, `iq1s_rows_avx2`) came from the Maya
history (PR #695, decode 31.0 -> 41.6 on DS4).  q36's CPU pool runs `native_quant_act` /
`native_expert_grouped` (ggml-cpu-based); only an IQ1_S Qwen pack would make this relevant (the Coder pack is
IQ1_M), and the native format may differ.  Treat as an experiment, not a port.

## Not applicable to q36

| DS4 change | Why |
| --- | --- |
| Peer prefill all-reduce, AR2, slab AR, fused attention+AR | q36 on two NVIDIA cards is a layer split; `q36_engine.cu` has no TP all-reduce |
| `DS4_ROUTE_FUSE` (gate/route GEMV pair) | q36's router is a native grouped MMQ, not DS4's F32 GEMV + route |
| `DS4_TP_SHARES` / TP balance | no TP FFN split in q36's layer-split mode |
| Cache substitution | q36 has `STRATA_ROUTE_PRIOR` of its own |
| Persistent copy pump | q36 already prefetches two layers ahead with events |

## Change points and references

```text
q36:  stream_kick            src/qwen35/q36_engine.cu ~2174
      rebalance / after_rows ~1282, ~2299
      warmup                 ~2758
      host arena             ~2924-2956 (cudaHostAlloc)
DS4:  skip-unpicked path     src/dsv41/ds4_model.cu ~4867 (skip_unpicked_rows, copy_ahead)
      huge-page arena        src/dsv41/ds4_model.cu (DS4_COPY_HUGETLB), tools/ds4_hugepages.sh
      measured log           DS4-MAYA-MERGE.md (sections "Short-prompt transfers", "Clean boot", "State gauge")
```

The DS4 work's own best numbers, for context only: 13K prefill 810 tok/s clean (Maya 1018), 6.6K at parity
(826 against 825), short prompt 2.3x, warm decode 37-53.  None of that is a Qwen claim.
