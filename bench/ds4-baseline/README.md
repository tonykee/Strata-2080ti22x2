# DS4 post-reboot baseline

Prompt IDs are saved here so a reboot (which may clear /tmp) does not lose them:
- prompt-3.5k.txt
- prompt-13k.txt

Run one engine at a time. Stop and release huge pages before starting the other.

## Maya v1 (tuned config; start + warmup)

```sh
cd /mnt/kioxia/maya
DS4_COPY_HUGETLB=1 DS4_COPY_PIN=1 DS4_SETTLE_MS=2000 ./start_ds4.sh
./.venv/bin/python tools/warmup_ds4.py --url http://127.0.0.1:8000 --key llama_local --rounds 2
./.venv/bin/python docs/2080ti22gx2/bench_ds4.py clean-v1
./stop_ds4.sh
```

Stop now releases the huge-page pool and compacts. Verify:

```sh
grep -E 'HugePages_Total|HugePages_Free|MemAvailable' /proc/meminfo
ss -ltn | grep ':8000 '
```

## Maya v2

```sh
cd /mnt/kioxia/maya
DS4_COPY_HUGETLB=1 DS4_COPY_PIN=1 DS4_SETTLE_MS=2000 ./start_ds4_v2.sh
./.venv/bin/python docs/2080ti22gx2/bench_ds4.py clean-v2
./stop_ds4_v2.sh
```

## Strata (current branch)

```sh
cd /home/likan/strata-latest
sudo -n /usr/local/sbin/strata-ds4-hugepages 96
IDS=$(cat bench/ds4-baseline/prompt-13k.txt)
DS4_ENGRAM_CACHE_MB=2048 DS4_SETTLE_MS=2000 DS4_IQ1S_ROWS=1 DS4_IQ_OWN=1 \
DS4_HIP_PEER_AR=1 DS4_STORE_GB=0 DS4_COPY_HUGETLB=1 DS4_COPY_PIN=1 DS4_ATTN_V7=1 \
./build-dsv41/strata-dsv41 --pack /mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-XS-v2/maya-ds4-v2 \
  --gpus 0,1 --tp 2 --max-context 32768 --reserve-mib 768 --cpu-threads 20 \
  --mmap-experts off --experts 4 --prefill-chunk 8192 --dspark on --ids "$IDS" --n 1
sudo -n /usr/local/sbin/strata-ds4-hugepages-release
```

Record:
- prefill tok/s (13K and 3.5K)
- decode 128 tok/s
- `HugePages_Free` during load
- `nvidia-smi` clocks/power/PCIe during load


## The fast state after a reboot (measured 2026-11-09)

The first two engine starts after a reboot run the 13K prompt at ~710 tok/s (load 53.8 s); the third
and later run ~560 tok/s (load 64-70 s), even with the huge-page pool kept, the ordinary THP arena,
a fixed expert profile, or the model page cache dropped. Only a reboot restores 710. Keep a server
running after a fresh boot for the fast state; `stop_cpu_lane.sh` now keeps the pool by default and
takes `--release` when another model needs the RAM.
