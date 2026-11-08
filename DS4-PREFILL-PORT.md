# DS4 CUDA prefill port (branch `ds4-cuda-prefill`)

Started from `ds4-cache-sub` (prefill direct-pinned DMA, 13K 449 tok/s).

## Findings

- The busy-experts FP16 GEMM path is already ported (`DS4_FAT_ROWS`).  With a full
  90.4 GB copy it now reaches `f16_rows` but that kernel hits an illegal memory access
  on the TP2 members: `run_set` passes the full FFN width while the source blob is a
  member's slice.  A `ringed` guard was added (Maya only runs FAT on VRAM experts),
  but the member width/offset still needs a per-member FAT port.  `DS4_FAT_ROWS`
  stays off; with `--reserve-mib 512` the old path ran but peer AR lost its 320 MiB
  and 13K prefill fell to 368 against 449 without FAT.
- Maya's `sparse_attn_v7_kernel` (Volta/Turing WMMA) needs 95,232 bytes of shared
  memory; the RTX 2080 Ti reports 65,536 bytes (`sharedMemPerBlockOptin`).  Maya's
  `attn_v7_now()` calls `cudaFuncSetAttribute` and falls back when that fails, so
  **Maya on this 2080 Ti pair does not use V7**; its 1,000 tok/s prefill comes from
  the other paths (FAT_ROWS, larger attention slabs, peer AR, scalar packed
  attention), not the V7 kernel.
- Maya's `sparse_attn_mma_kernel` is sm_80+ (Ampere); it compiles out on sm_75.
- Therefore the next port targets the **scalar/packed attention and FAT_ROWS**
  paths Maya actually uses on 2080 Ti, not V7.

## Next port options

1. Split the V7 attention tile (smaller row tile / high-only Q) to fit 64 KiB, or
   use a two-pass shared-memory scheme.
2. Port the newer scalar packed attention improvements from Maya, which do not need
   tensor cores.
3. Debug the busy-experts cublas status 13 with a minimal sm_75 GEMM test before
   enabling `DS4_FAT_ROWS`.

The current safe result is **532 tok/s at 13K** after porting Maya's multi-row
router GEMM (`gemv_f32_rows_kernel`): the FFN mHC + router + sort stage fell from
9.5 s to 2.9 s.  A 64 KiB sm_75 V7 attention variant (`DS4_ATTN_V7=1`) now runs:
85.8 s sparse attention -> 8.6 s, best measured 13K **584.7 tok/s**; high-only Q is
approximate (quality still to verify).  With `DS4_SKIP_UNPICKED=1` it measured 562.9
(11,840 copied / 38,880 left out).  `DS4_ATTN_V7` stays opt-in.  `DS4_ATTN_G` is
available for heads-per-block A/B (4 slower; 0/8 default).  The fixed 2048-row slab
helps 13K but hurts 3.5K, so the server config keeps the automatic slab.  FAT stays
off for TP2 until the member layout is ported.
