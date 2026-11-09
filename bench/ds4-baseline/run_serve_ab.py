#!/usr/bin/env python3
"""Two prompts through one engine process: A cold, then B with the expert cache from A warm.

A and B are different 13K token sequences (no shared prefix), so B re-reads its whole prompt while
the VRAM expert slots the first read landed (DS4_LAND / DS4_RECENT) stay.  Prints the engine's own
prompt/decode lines for both.  Env overrides from the shell apply to the engine.
"""
import json
import os
import subprocess
import sys
import threading

ROOT = '/home/likan/strata-latest'
BASE = ROOT + '/bench/ds4-baseline/'
PACK = '/mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-XS-hybrid/maya-ds4-hybrid'
PROFILE = '/mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-XS-v2/maya-ds4-v2/expert-profile.bin'

ids_a = open(BASE + 'prompt-13k-a.txt').read().strip()
ids_b = open(BASE + 'prompt-13k-b.txt').read().strip()

cmd = [ROOT + '/build-dsv41/strata-dsv41', '--pack', PACK,
       '--expert-profile', PROFILE, '--profile-readonly',
       '--gpus', '0,1', '--tp', '2', '--max-context', '32768', '--reserve-mib', '768',
       '--cpu-threads', '20', '--mmap-experts', 'off', '--experts', '4', '--prefill-chunk', '8192',
       '--dspark', 'on', '--serve']
print('env:', {k: v for k, v in os.environ.items() if k.startswith('DS4_')}, flush=True)
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     text=True, bufsize=1, cwd=ROOT)
interesting = ('prompt:', 'decode:', 'recent', 'in VRAM', 'copied into RAM', 'one-GPU',
               'prefill host', 'prefill streams', 'CUDA0 layers', 'CUDA1 layers', 'ring')


def err():
    for ln in p.stderr:
        if any(k in ln for k in interesting):
            print('ERR', ln.rstrip(), flush=True)


threading.Thread(target=err, daemon=True).start()

while True:
    ln = p.stdout.readline()
    if not ln:
        print('EOF before READY', flush=True)
        sys.exit(1)
    if ln.startswith('READY'):
        print('READY', ln.strip(), flush=True)
        break


def gen(tag, ids, n):
    p.stdin.write('GEN %d %s\n' % (n, ids))
    p.stdin.flush()
    out = []
    while True:
        ln = p.stdout.readline()
        if not ln:
            print('EOF', tag, flush=True)
            return out
        if ln.startswith('DONE') or ln.startswith('ERR'):
            print(tag, ln.strip(), flush=True)
            return out
        out.append(ln)


gen('A', ids_a, 16)
gen('B', ids_b, 16)
p.stdin.write('QUIT\n')
p.stdin.flush()
p.wait(timeout=60)
print('rc', p.returncode, flush=True)
