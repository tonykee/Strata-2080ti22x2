#!/usr/bin/env python3
"""warmup_ds4.py - exercise DS4 after a start so the first real request is not the slow one.

A DS4 start lays the experts out from the pack's profile; the first prompts and
decodes then teach the VRAM expert cache and the prompt path their working set.
Until that has happened, a long prompt can spend seconds staging experts and a
decode can run at roughly half speed.  This sends a short decode, a 1.6K and a
13K prompt (two chunks: the prompt-ring path) and a decode, then prints the
engine's vram_hit/cpu if /metrics is reachable.

Placement only: the model's answers are not changed by the warm-up.

Usage:
  .venv/bin/python tools/warmup_ds4.py [--url URL] [--key KEY] [--rounds N]

The start scripts run it once after /health is ready (MAYA_WARMUP=0 skips it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = ("混合专家模型通过稀疏激活在保持参数规模的同时控制计算量。每个词元只经过少数专家，"
        "路由网络决定哪些专家参与计算，其余专家保持空闲。本地部署时专家分层缓存在显存、内存和固态硬盘之间，"
        "按对话中出现的频率动态迁移。工程实践中需要专家并行、容量因子和丢词策略的配合。")


def request(url: str, key: str, text: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "off",
    }).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def prompt(n: int) -> str:
    text = f"[{uuid.uuid4().hex}] 请阅读下面材料，最后只回答“收到”。\n\n"
    return text + BASE * max(1, int(n * 1.4 / len(BASE)))


def metrics(url: str, key: str) -> dict | None:
    try:
        headers = {"Authorization": "Bearer " + key} if key else {}
        req = urllib.request.Request(url.rstrip("/") + "/metrics", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return (json.load(r).get("tiers") or {}).get("now")
    except Exception:
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="")
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args(argv[1:])

    try:
        t0 = time.time()
        d = request(args.url, args.key, "热身：只回答“好”。", 32)
        print(f"  warmup short decode: {d['timings']['predicted_per_second']:.1f} tok/s", flush=True)
        for _ in range(max(1, args.rounds)):
            for n in (1600, 13000):
                d = request(args.url, args.key, prompt(n), 8)
                t = d["timings"]
                print(f"  warmup prefill {d['usage']['prompt_tokens']:>6}: "
                      f"{t['prompt_per_second']:7.1f} tok/s  {t['prompt_ms'] / 1000:5.2f}s", flush=True)
            d = request(args.url, args.key, prompt(0) + "写一段 128 token 的说明。", 128)
            print(f"  warmup decode: {d['timings']['predicted_per_second']:.1f} tok/s", flush=True)
        m = metrics(args.url, args.key)
        if m:
            print("  warmup state: " + " ".join(
                f"{k}={m[k]}" for k in ("tok_s", "vram_hit", "cpu", "ram_fetch", "engram_ms") if k in m), flush=True)
        print(f"  warmup done in {time.time() - t0:.0f}s", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001 - a warm-up is best effort: never keep the server from starting
        print(f"  warmup skipped: {e}", file=sys.stderr, flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
