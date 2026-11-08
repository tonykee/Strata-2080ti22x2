# DS4 CUDA prefill port (branch `ds4-cuda-prefill`)

Started from `ds4-cache-sub` (prefill direct-pinned DMA, 13K 449 tok/s).

## Findings

- The busy-experts FP16 GEMM path is already ported (`DS4_FAT_ROWS`).  With a full
  90.4 GB copy and `--reserve-mib 768` it returned cublas status 13 (execution failed;
  the cublas workspace no longer fit in CUDA0's 206 MiB free).  With
  `--reserve-mib 512` it runs, but peer AR then fails to allocate and 13K prefill is
  368 tok/s against 449 without FAT; not adopted.  `DS4_FAT_ROWS` stays off.
- Maya's `sparse_attn_v7_kernel` (Volta/Turing WMMA) needs 95,232 bytes of shared
  memory; the RTX 2080 Ti reports 65,536 bytes (`sharedMemPerBlockOptin`).  The
  kernel cannot be launched unchanged on this pair.
- Maya's `sparse_attn_mma_kernel` is sm_80+ (Ampere); it compiles out on sm_75.

## Next port options

1. Split the V7 attention tile (smaller row tile / high-only Q) to fit 64 KiB, or
   use a two-pass shared-memory scheme.
2. Port the newer scalar packed attention improvements from Maya, which do not need
   tensor cores.
3. Debug the busy-experts cublas status 13 with a minimal sm_75 GEMM test before
   enabling `DS4_FAT_ROWS`.

The current safe result remains the direct-pinned prefill path:
13K 449 tok/s, 3.5K 268-331 tok/s depending on the opt-in skip.
