import json, time, urllib.request, os, sys, threading
import os
KEY = os.environ.get("API_KEY") or open("/root/.api_key").read().strip(); M = os.environ.get("MODEL", "glm-5.3-flash")
N = int(sys.argv[1]); TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 400
temas = ["cache de prefixo","decodificacao especulativa","atencao esparsa","quantizacao NVFP4","paralelismo de tensores","expert parallelism","CUDA graphs","fragmentacao de memoria","NCCL em PCIe","KV cache FP8","chunked prefill","escalonador FCFS","tokenizacao BPE","RoPE","MoE com 384 experts","offload de experts"]
out = [0]*N; lock = threading.Lock()
def run(i):
    body = {"model": M, "messages": [{"role": "user", "content": f"Escreva um texto tecnico detalhado sobre {temas[i%len(temas)]} em inferencia de LLM (variante {i})."}], "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.7, "max_tokens": TOK}
    r = urllib.request.Request("" + os.environ.get("BASE", "http://127.0.0.1:8000") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=900).read()); out[i] = d["usage"]["completion_tokens"]
for rep in range(2):
    ts = [threading.Thread(target=run, args=(i,)) for i in range(N)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; dt = time.time() - t0
    print(f"{N} fluxos x {TOK}: {sum(out)} tokens em {dt:.1f}s = {sum(out)/dt:.0f} tok/s agregado ({sum(out)/dt/N:.0f} por fluxo)", flush=True)
