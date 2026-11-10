#!/usr/bin/env python3
"""Standard Swift benchmark: prefill + 512-token decode at 20K/50K/100K.

Same protocol as OPT-2026-10.md 9.28/9.29: one user message, about
20K/50K/100K tokens, max_tokens=512, temperature=0.  The engine log is the
source of truth for prefill tok/s and decode tok/s.

Record copy: ROOT, TOKDIR and ENGINE_LOG are the local 2080 Ti paths from
2026-11-10; adjust them for another tree.  The script expects a server
already running at port 8000 (start it with `strata serve --profile ...`).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = pathlib.Path("/home/likan/strata-latest")
TOKDIR = pathlib.Path("/home/likan/Strata-data/packs/swift-iq3_s/tokenizer")
ENGINE_LOG = ROOT / "strata-swift-latest.log"
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODELS = "http://127.0.0.1:8000/v1/models"
KEY = "llama_local"

SENTENCE = ("During the monitoring window the inference engine streamed the expert "
            "weights from host memory across the PCIe link while both graphics cards "
            "kept their own layer ranges resident.")
INSTRUCTION = ("\n\nList the integers from 1 to 500, one per line, with no other text.")


def load_tokenizer():
    sys.path.insert(0, str(ROOT / "tools"))
    import strata_tokenizer as ST
    vocab = json.loads((TOKDIR / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (TOKDIR / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((TOKDIR / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types)


def make_content(tok, target_tokens: int, nonce: str) -> str:
    """Filler sized by the tokenizer, then the fixed 512-token instruction."""
    tail_ids = tok.encode(INSTRUCTION)
    need = max(target_tokens - len(tail_ids), 16)
    lines = []
    i = 1
    while True:
        lines.append("segment %d ref %08x: %s" % (i, (i * 2654435761) & 0xFFFFFFFF, SENTENCE))
        i += 1
        text = "\n".join(lines)
        if len(text) > need * 5 + 2000:      # ~4.5 chars/token; trimming is exact anyway
            break
    ids = tok.encode(text)[:need]
    filler = tok.decode(ids)
    assert tok.encode(filler) == ids
    return "run %s:\n%s%s" % (nonce, filler, INSTRUCTION)


def model_id() -> str:
    req = urllib.request.Request(MODELS, headers={"Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        obj = json.loads(r.read())
    return obj["data"][0]["id"]


def post(content: str, model: str, max_tokens: int, timeout: float = 1800):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out, time.monotonic() - t0


PROMPT_RE = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([0-9.]+) ms "
    r"\(([0-9.]+) tok/s\), (\d+) generated in ([0-9.]+) ms \(([0-9.]+) tok/s\), "
    r"drafts accepted (\d+) of (\d+)")
CACHE_RE = re.compile(r"decode expert cache hit rate: ([0-9.]+)%.*?\((\d+) hits / (\d+) lookups\)")
KV_RE = re.compile(r"KV streaming: ([0-9.]+)% of (\d+) block reads hit VRAM, ([0-9.]+) MiB read from RAM")


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.samples = []

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,power.draw,clocks.sm,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], text=True, timeout=5)
                for line in out.strip().splitlines():
                    p = [x.strip() for x in line.split(",")]
                    self.samples.append({
                        "t": time.time(), "gpu": int(p[0]), "power_w": float(p[1]),
                        "sm_mhz": float(p[2]), "util_pct": float(p[3]), "mem_mib": float(p[4])})
            except Exception:
                pass
            self.stop.wait(1.0)


def wait_ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(MODELS, headers={"Authorization": "Bearer " + KEY})
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server did not become ready")


def new_lines(offset):
    try:
        lines = ENGINE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return lines[offset:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--lengths", default="20000,50000,100000")
    ap.add_argument("--tag", default="x99")
    ap.add_argument("--out", default="/tmp/opencode/swift-x99-bench.json")
    ap.add_argument("--warm", action="store_true", default=True)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    tok = load_tokenizer()
    ready_s = wait_ready()
    model = model_id()
    print("server ready in %.0fs; model=%s" % (ready_s, model), flush=True)

    sampler = GpuSampler()
    sampler.start()
    results = []

    def run_one(label, length, max_tokens):
        nonce = uuid.uuid4().hex[:12]
        content = make_content(tok, length, nonce)
        ids = tok.encode(content)
        offset = len(ENGINE_LOG.read_text(encoding="utf-8", errors="replace").splitlines())
        t0 = time.time()
        out, wall = post(content, model, max_tokens)
        lines = new_lines(offset)
        prompt = cache = kv = None
        for line in lines:
            m = PROMPT_RE.search(line)
            if m:
                prompt = {
                    "prompt_tokens": int(m.group(1)), "reused": int(m.group(2)),
                    "read": int(m.group(3)), "prefill_ms": float(m.group(4)),
                    "prefill_tps": float(m.group(5)), "generated": int(m.group(6)),
                    "decode_ms": float(m.group(7)), "decode_tps": float(m.group(8)),
                    "draft_accepted": int(m.group(9)), "draft_total": int(m.group(10)),
                }
            m = CACHE_RE.search(line)
            if m:
                cache = {"hit_pct": float(m.group(1)), "hits": int(m.group(2)), "lookups": int(m.group(3))}
            m = KV_RE.search(line)
            if m:
                kv = {"vram_pct": float(m.group(1)), "reads": int(m.group(2)), "ram_mib": float(m.group(3))}
        usage = out.get("usage", {})
        rec = {
            "label": label, "target_tokens": length, "content_tokens": len(ids),
            "wall_s": round(wall, 3), "usage": usage, "prompt": prompt,
            "expert_cache": cache, "kv": kv,
            "reply_chars": len(((out.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""),
            "t": t0,
        }
        results.append(rec)
        if prompt:
            print("%-18s target %6d actual %6d reused %5d | prefill %7.1f tok/s "
                  "(%6.1f ms) | decode %6.1f tok/s (%4d tok, %5.1f ms) | draft %d/%d | "
                  "cache %s | wall %.1fs"
                  % (label, length, prompt["prompt_tokens"], prompt["reused"],
                     prompt["prefill_tps"], prompt["prefill_ms"],
                     prompt["decode_tps"], prompt["generated"], prompt["decode_ms"],
                     prompt["draft_accepted"], prompt["draft_total"],
                     (cache or {}).get("hit_pct"), wall), flush=True)
        else:
            print("%-18s NO PROMPT LINE (wall %.1fs)" % (label, wall), flush=True)
            for line in lines[-8:]:
                print("   | " + line, flush=True)
        return rec

    if args.warm:
        print("warmup ...", flush=True)
        run_one("warmup-short", 400, 32)
        run_one("warmup-20k", 20000, 32)

    for rep in range(args.reps):
        for length in lengths:
            run_one("r%d-%dK" % (rep + 1, length // 1000), length, 512)

    sampler.stop.set()
    sampler.join(timeout=3)

    # summary
    print("\n=== summary (%s) ===" % args.tag, flush=True)
    summary = {}
    for length in lengths:
        rows = [r["prompt"] for r in results
                if r["prompt"] and r["target_tokens"] == length and r["label"].startswith("r")]
        if not rows:
            continue
        pf = [r["prefill_tps"] for r in rows]
        dc = [r["decode_tps"] for r in rows]
        acc = [100.0 * r["draft_accepted"] / r["draft_total"] if r["draft_total"] else 0 for r in rows]
        summary[str(length)] = {
            "n": len(rows),
            "prefill_mean": statistics.mean(pf), "prefill_sd": statistics.stdev(pf) if len(pf) > 1 else 0,
            "decode_mean": statistics.mean(dc), "decode_sd": statistics.stdev(dc) if len(dc) > 1 else 0,
            "accept_mean": statistics.mean(acc),
            "prompt_tokens_mean": statistics.mean([r["prompt_tokens"] for r in rows]),
        }
        s = summary[str(length)]
        print("%3dK: prefill %7.1f +/- %5.1f tok/s | decode %5.1f +/- %4.1f tok/s | "
              "accept %5.1f%% | n=%d" % (length // 1000, s["prefill_mean"], s["prefill_sd"],
                                         s["decode_mean"], s["decode_sd"], s["accept_mean"], s["n"]))

    # GPU stats while a request is running (util > 20)
    busy = [s for s in sampler.samples if s["util_pct"] > 20]
    gpu = {}
    for g in sorted({s["gpu"] for s in busy}):
        rows = [s for s in busy if s["gpu"] == g]
        gpu[g] = {
            "n": len(rows),
            "power_mean": statistics.mean([s["power_w"] for s in rows]),
            "power_max": max(s["power_w"] for s in rows),
            "sm_mean": statistics.mean([s["sm_mhz"] for s in rows]),
            "sm_max": max(s["sm_mhz"] for s in rows),
            "util_mean": statistics.mean([s["util_pct"] for s in rows]),
            "mem_max": max(s["mem_mib"] for s in rows),
        }
    print("GPU busy: " + json.dumps(gpu, ensure_ascii=False), flush=True)

    payload = {"tag": args.tag, "model": model, "reps": args.reps, "summary": summary,
               "gpu": gpu, "results": results, "samples": sampler.samples}
    pathlib.Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print("written " + args.out, flush=True)


if __name__ == "__main__":
    main()
