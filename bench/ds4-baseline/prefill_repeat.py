#!/usr/bin/env python3
"""Maya's bench prompt (a paragraph repeated to N tokens) through the API, prefill only."""
import json
import sys
import time
import urllib.request
import uuid

URL = "http://127.0.0.1:8000/v1/chat/completions"
KEY = "llama_local"
BASE = ("混合专家模型通过稀疏激活在保持参数规模的同时控制计算量。每个词元只经过少数专家，"
        "路由网络决定哪些专家参与计算，其余专家保持空闲。本地部署时专家分层缓存在显存、内存和固态硬盘之间，"
        "按对话中出现的频率动态迁移。工程实践中需要专家并行、容量因子和丢词策略的配合。")


def post(text, max_tokens=8):
    body = json.dumps({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "off"}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": f"Bearer {KEY}"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    tt = d["timings"]
    u = d["usage"]
    print(f"prompt={u['prompt_tokens']} cached={u['prompt_tokens_details']['cached_tokens']} "
          f"prefill={tt['prompt_ms']/1000:.2f}s {tt['prompt_per_second']:.1f} tok/s wall={time.time()-t:.1f}s",
          flush=True)


post("热身：只回答“好”。", 16)
for n in (2000, 8000, 16000):
    text = "[" + uuid.uuid4().hex + "] 请阅读下面材料，最后只回答“收到”。\n\n" + BASE * int(n * 1.4 / len(BASE))
    post(text)
