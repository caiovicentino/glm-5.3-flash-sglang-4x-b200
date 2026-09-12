import json, time, urllib.request, os
import os
KEY = os.environ.get("API_KEY") or open("/root/.api_key").read().strip(); M = os.environ.get("MODEL", "glm-5.3-flash")
tarefas = ["um cache LRU com lista duplamente ligada e dicionario, com get, put, delete, iteracao e docstrings",
           "um parser de CSV com aspas, escapes e streaming linha a linha, com testes unittest",
           "um rate limiter token bucket thread-safe com decorator e testes",
           "um cliente HTTP com retry exponencial, jitter e circuit breaker, tipado"]
t0 = time.time(); tot = 0
for t in tarefas:
    body = {"model": M, "max_tokens": 700, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Escreva em Python {t}. Retorne so codigo."}]}
    r = urllib.request.Request("" + os.environ.get("BASE", "http://127.0.0.1:8000") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    tot += json.loads(urllib.request.urlopen(r, timeout=900).read())["usage"]["completion_tokens"]
print(f"  codigo: {tot} tokens em {time.time()-t0:.0f}s = {tot/(time.time()-t0):.0f} tok/s")
