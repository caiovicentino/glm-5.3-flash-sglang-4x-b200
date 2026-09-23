# ops/ — changing a production server without a human watching the boot

`swap.sh` stops the running server, starts a profile from `serve/profiles/`, and only keeps it if three things
hold; otherwise it starts the old profile again by itself.

1. **It comes up** — `/v1/models` answers 200 within `BOOT_TIMEOUT`.
2. **The memory guard passes** — read from the boot log: KV pool ≥ `MIN_KV_TOKENS`, free memory on TP0 after
   CUDA-graph capture ≥ `MIN_FREE_GB`, and, if set, the KDA slot count equals `MAMBA_SLOTS_EXPECT`. Free memory
   after graph capture is the number that predicts request-level OOMs on this model: 36.70 GB (v2) and 37.74 GB
   (v3) were fine; ~25 GB (the first EP4 attempt) was not.
3. **`smoke.py` passes** — models, reasoning split, streaming, tool call, vision, prefix cache + cache report,
   `/metrics`, 16 parallel requests, pt-BR generation without degeneration; `SMOKE_IMAGE_CPU=1` adds "the HTTP
   process does not grow on the GPU after 12 images" as mandatory (the v3.1 fix).

On 2026-09-23 it rolled production back by itself when `--tokenizer-worker-num 4` killed the boot (that flag is
incompatible with `--api-key`): 4 min 37 s from stop to the old profile serving again. Nine hours later the same
machinery reverted v3 after the monitoring caught the image pre-processing OOM (2 min 16 s).

```bash
NEW=v3.1 OLD=v2 MAMBA_SLOTS_EXPECT=700 SMOKE_IMAGE_CPU=1 \
SMOKE_EXPECT="prefill_decode_interval=4,cuda_graph_max_bs_decode=96,max_mamba_cache_size=700,image_processor_backend=pil" \
  setsid nohup bash ops/swap.sh >/dev/null 2>&1 </dev/null &
tail -f /root/swap.log
```

Lessons that cost us time, so the script avoids them:

- **Validate flags by parsing first, but know what that misses.** `prepare_server_args(argv)` catches invalid
  combinations without touching the GPUs, but not assertions raised during HTTP startup (the multi-tokenizer +
  API-key one). Only a real boot with rollback catches those.
- **Never probe the running server's GPUs from a new CUDA context** during a dry run: GPUs 1–3 sit at ~0.5 GB
  free under steady load, and a fresh context can push the server into OOM. Run dry runs with
  `CUDA_VISIBLE_DEVICES=0` (GPU 0 had the most headroom) or with no GPU at all.
- **`pgrep -f` lies inside `ssh 'command'`**: it matches the SSH command line itself, and the bracket trick
  (`[l]aunch_server`) only protects you if the plain string appears nowhere else in that line. `swap.sh` runs as
  its own file for this reason. To cancel a scheduled `bash -c "sleep N; cmd"`, kill the wrapper by exact PID
  (from `ps -eo pid,ppid,args`) before the `sleep`, and never kill a computed parent PID without checking it is
  not 1.
