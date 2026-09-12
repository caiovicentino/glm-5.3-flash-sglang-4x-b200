"""TTFT por tamanho de prompt (prefill frio, sem cache): prompts únicos por salt, streaming, mede o 1º token."""
import os
import json, time, urllib.request, os, random, string
KEY = os.environ.get("API_KEY") or open("/root/.api_key").read().strip(); M = os.environ.get("MODEL", "glm-5.3-flash")
base = open(os.environ.get("TTFT_SOURCE", "/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py")).read()  # any long text file works
def prompt(n_chars):
    salt = "".join(random.choices(string.ascii_lowercase, k=12))
    return f"[{salt}] Leia o codigo abaixo e responda em uma frase o que a classe principal faz.\n\n{base[:n_chars]}\n\nResposta em uma frase:"
def ttft(p):
    body = {"model": M, "messages": [{"role": "user", "content": p}], "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "max_tokens": 24, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request("" + os.environ.get("BASE", "http://127.0.0.1:8000") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time(); first = None; usage = None; txt = ""
    with urllib.request.urlopen(r, timeout=900) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if first is None: first = time.time() - t0
                txt += ch[0]["delta"]["content"]
    return first, usage["prompt_tokens"] if usage else -1, txt
for n in [4000, 16000, 32000, 64000, 120000]:
    f, pt, txt = ttft(prompt(n))
    print(f"prompt={pt:6d} tok  TTFT={f:6.2f}s  prefill≈{pt/f:6.0f} tok/s :: {txt[:70]!r}", flush=True)
