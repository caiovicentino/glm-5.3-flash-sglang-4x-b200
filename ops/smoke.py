#!/usr/bin/env python3
"""Smoke tests for an SGLang GLM-5.3-Flash server, run right after a (re)start.

Exit code 0 when every mandatory test passes. Configuration via environment:
  SMOKE_BASE        server base URL (default http://127.0.0.1:8000)
  SMOKE_MODEL       served model name (default glm-5.3-flash)
  API_KEY           API key (default: contents of /root/.api_key)
  SMOKE_EXPECT      comma-separated server_info expectations, e.g.
                    "prefill_decode_interval=4,cuda_graph_max_bs_decode=96" (checked, not mandatory)
  SMOKE_IMAGE_CPU   "1" makes the "image pre-processing stays off the GPU" check mandatory
"""
import base64, io, json, os, random, re, subprocess, sys, time, urllib.request
import concurrent.futures as cf

B = os.environ.get("SMOKE_BASE", "http://127.0.0.1:8000").rstrip("/")
M = os.environ.get("SMOKE_MODEL", "glm-5.3-flash")
K = os.environ.get("API_KEY") or open("/root/.api_key").read().strip()
LOW = {"reasoning_effort": "low"}
res = []


def http(path, body=None, timeout=300, auth=True):
    h = {"Content-Type": "application/json"}
    if auth:
        h["Authorization"] = "Bearer " + K
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(B + path, data=data, headers=h), timeout=timeout)


def chat(msgs, **kw):
    return json.loads(http("/v1/chat/completions", {"model": M, "messages": msgs, **kw}).read())


def teste(nome, fn, obrig=True):
    t0 = time.time()
    try:
        ok, det = fn()
    except Exception as e:  # noqa: BLE001 — any failure is a failed test
        ok, det = False, f"{type(e).__name__}: {str(e)[:180]}"
    res.append((nome, ok, obrig))
    tag = "OK    " if ok else ("FAILED" if obrig else "WARN  ")
    print(f"  {tag} {nome} ({time.time() - t0:.1f}s) {det}", flush=True)


def entry_process_gpu_mib():
    """GPU memory held by the HTTP/tokenizer process (python -m sglang.launch_server), 0 if none."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    tot = 0
    for line in out.strip().splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "sglang.launch_server" in cmd:
            tot += int(mem)
    return tot


def png_url(im):
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def t_models():
    ids = [m["id"] for m in json.loads(http("/v1/models").read())["data"]]
    return M in ids, str(ids)


def t_reasoning():
    d = chat([{"role": "user", "content": "Quanto é 17*23? Responda só o número."}], max_tokens=600, chat_template_kwargs=LOW)
    m = d["choices"][0]["message"]; c = m.get("content") or ""
    return ("391" in c and "think>" not in c), f"content={c[:40]!r} reasoning={len(m.get('reasoning_content') or '')}ch"


def t_stream():
    b = {"model": M, "messages": [{"role": "user", "content": "Diga uma frase curta sobre o mar."}],
         "max_tokens": 300, "stream": True, "chat_template_kwargs": LOW}
    n, ct = 0, ""
    with http("/v1/chat/completions", b) as r:
        for raw in r:
            if not raw.startswith(b"data: ") or b"[DONE]" in raw:
                continue
            ch = json.loads(raw[6:]).get("choices") or []
            if ch:
                n += 1; ct += (ch[0].get("delta") or {}).get("content") or ""
    return (n >= 3 and len(ct) > 5 and "think>" not in ct), f"chunks={n} content={ct[:50]!r}"


def t_tool():
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Clima atual de uma cidade",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    d = chat([{"role": "user", "content": "Qual o clima agora em Recife? Use a ferramenta."}],
             tools=tools, tool_choice="auto", max_tokens=800, chat_template_kwargs=LOW)
    tc = d["choices"][0]["message"].get("tool_calls") or []
    if not tc:
        return False, "no tool_calls"
    f = tc[0]["function"]; a = json.loads(f["arguments"])
    return (f["name"] == "get_weather" and "recife" in str(a).lower()), f"{f['name']}({a})"


def t_vision():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (320, 320), "white"); ImageDraw.Draw(im).ellipse((60, 60, 260, 260), fill=(220, 20, 20))
    d = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_url(im)}},
              {"type": "text", "text": "Qual é a cor do círculo? Responda com uma palavra."}]}],
             max_tokens=400, chat_template_kwargs=LOW)
    c = (d["choices"][0]["message"].get("content") or "").lower()
    return ("vermelh" in c or "red" in c), f"content={c[:40]!r}"


def t_images_off_gpu(before):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(int(time.time()))
    for w, h in [(640, 480), (1024, 768), (1280, 720), (1600, 1200), (1920, 1080), (2048, 1536)] * 2:
        im = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_url(im)}},
              {"type": "text", "text": "Responda só: ok"}]}], max_tokens=50, chat_template_kwargs=LOW)
    after = entry_process_gpu_mib()
    return (after - before < 64), f"entry process on GPU: {before} → {after} MiB after 12 images"


def t_cache():
    salt = f"[smoke {random.randint(0, 10**12)}] "
    msgs = [{"role": "user", "content": salt + ("O cache de prefixo guarda o estado de trechos já vistos. " * 900) + "\nResponda só: ok"}]
    chat(msgs, max_tokens=20, chat_template_kwargs=LOW)
    d = chat(msgs, max_tokens=20, chat_template_kwargs=LOW)
    pt = d["usage"]["prompt_tokens"]; c = (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return (pt > 5000 and c >= 0.8 * pt), f"prompt={pt} cached_on_2nd={c}"


def t_metrics():
    t = http("/metrics", auth=False).read().decode()
    return ("sglang:num_running_reqs" in t), f"{len(t)} bytes"


def t_parallel():
    def one(i):
        d = chat([{"role": "user", "content": f"Diga o número {i} por extenso."}], max_tokens=300, chat_template_kwargs=LOW)
        return bool((d["choices"][0]["message"].get("content") or "").strip())
    with cf.ThreadPoolExecutor(16) as ex:
        r = list(ex.map(one, range(16)))
    return all(r), f"{sum(r)}/16 with content"


def t_ptbr():
    d = chat([{"role": "user", "content": "Explique em três parágrafos, em português, por que o cache de prefixo acelera a inferência de modelos de linguagem."}],
             max_tokens=2500, chat_template_kwargs=LOW)
    c = d["choices"][0]["message"].get("content") or ""; w = re.findall(r"\w+", c.lower())
    degen = bool(re.search(r"(?:\b[oOaAeE]\b[ :.,]{1,2}){6,}", c)); uniq = len(set(w)) / max(len(w), 1)
    return (len(c) > 600 and not degen and uniq > 0.3), f"{len(c)} chars, unique words {uniq:.0%}, degenerate={degen}"


def t_info():
    exp = os.environ.get("SMOKE_EXPECT", "")
    d = json.loads(http("/get_server_info").read())
    got, bad = {}, []
    for kv in filter(None, exp.split(",")):
        k, v = kv.split("=")
        got[k] = d.get(k)
        if str(d.get(k)) != v:
            bad.append(k)
    return (not bad), (str(got) if got else "no expectations set")


def t_health():
    t0 = time.time(); http("/health", auth=False).read()
    return True, f"/health in {time.time() - t0:.3f}s"


gpu0 = entry_process_gpu_mib()                       # before any image request
teste("models", t_models)
teste("reasoning split", t_reasoning)
teste("streaming", t_stream)
teste("tool call", t_tool)
teste("vision", t_vision)
teste("image pre-processing off the GPU", lambda: t_images_off_gpu(gpu0), obrig=os.environ.get("SMOKE_IMAGE_CPU") == "1")
teste("prefix cache + cache report", t_cache)
teste("/metrics", t_metrics)
teste("16 in parallel", t_parallel)
teste("pt-BR, no degeneration", t_ptbr)
teste("server_info expectations", t_info, obrig=False)
teste("/health latency", t_health, obrig=False)
fail = [n for n, ok, ob in res if ob and not ok]
print("SMOKE_OK" if not fail else f"SMOKE_FAILED: {fail}", flush=True)
sys.exit(0 if not fail else 1)
