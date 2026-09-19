# GLM-5.3-Flash-NVFP4 on 4× NVIDIA B200 with SGLang

The recipe we validated on 2026-09-11/12 for serving **GLM-5.3-Flash-NVFP4** (320B total / 18B active,
vision, DSA sparse attention, MTP speculative decoding) on **four B200** (SM100, 183 GB each, NVLink)
with SGLang, tuned for **high concurrency**: 64 to 128 simultaneous streams. It is the third box in
a series measured with the same scripts:

- [glm-5.3-flash-sglang-4x-rtx-pro-6000](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-rtx-pro-6000) — production, SM120, PCIe
- [glm-5.3-flash-sglang-2x-b200](https://github.com/caiovicentino/glm-5.3-flash-sglang-2x-b200) — 2× B200, TP2
- this repo — 4× B200, TP4

Short version: **up to 24 streams, 4× B200 in TP4 is no faster than 2× B200 in TP2** (per-step latency
bound, same ~2.4k tok/s aggregate). The four cards pay off above 48 streams: **4.9k tok/s at 64
streams and 6.0k tok/s at 96**, 11× the ceiling of the SM120 production box, with a 7–10 M-token
fp8 KV pool. The one change that unlocks it is the adaptive-speculation config with one candidate per
batch bucket (`serve/adaptive_spec.json`): the stock adaptive policy switches drafting off at batch
64 and loses 40 %.

## Recipe card

| | |
|---|---|
| hardware | 4× B200 183 GB, NVLink NV18 all-to-all, Xeon Platinum 8559C 192 vCPU, 2 TB RAM, 1 TB NVMe — a Vast.ai verified-datacenter offer at ~US$ 28/h |
| image | `lmsysorg/sglang:glm-5.3-flash` (SGLang `0.0.0.dev1+gfe236ea6c3.mmfix1`, flashinfer 0.6.17, torch 2.13+cu130, driver 595). Launch the instance from this image; nothing else to install |
| checkpoint | [`RadixArk/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4) (190 GB, ModelOpt 0.46, the one the SGLang cookbook uses): routed experts + MLP NVFP4 g16, attention / router gates / embeddings / lm_head / vision tower / MTP layer BF16 |
| patches | **none** — the cookbook checkpoint loads on the stock image. The 0xSero loader patches that the 2× B200 recipe needs break the import on this newer build and are not required |
| chat template | the checkpoint's own (`emit_image` macro present, vision works out of the box) |
| parallelism | `--tp-size 4 --ep-size 4` — **EP4 added 2026-09-18**; the SGLang cookbook recipe uses it and we had been running TP-only with 288 experts. See "Update" below |
| attention | `--dsa-prefill-backend trtllm --dsa-decode-backend trtllm --kv-cache-dtype fp8_e4m3` (the Blackwell pair from the cookbook); vision tower on `fa4` (auto) |
| MoE | `--moe-runner-backend flashinfer_cutlass` with `--quantization modelopt_fp4` |
| speculation | `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --speculative-adaptive`, config `serve/adaptive_spec.json`: 5 steps at batch 1, 3 at batch ≥ 2, 2 at batch ≥ 40 |
| memory | `--mem-fraction-static 0.75` → KV pool **6.3 M tokens**, ~21 GiB free per card. **Do not raise this with a long context** — the DSA indexer buffer is `chunk × ctx × 4B` and comes out of the free memory, not the pool (incidents 8 and 9) |
| prefill | `--chunked-prefill-size 4096 --max-prefill-tokens 4096` — 8192 with a long context is what killed the engine on 09-14 (incident 8) |
| concurrency | `--max-running-requests 96 --cuda-graph-max-bs-decode 60`. Graphs to 60 cover the real traffic (largest batch seen: 41); larger batches are still accepted, they just run ungraphed |
| context | `--context-length 524288`. Raised 2026-09-18 because production was **rejecting real requests** at 373k–390k tokens. It is not free: the indexer buffer scales with it (incident 8) |
| client | OpenAI-compatible on :8000, `--reasoning-parser glm45 --tool-call-parser glm47`, `--enable-multimodal`, `--enable-cache-report`, `--enable-metrics` |

## Update 2026-09-18/19 — EP4, 524k context, and the memory that pays for both

Eight days of continuous production later, three things changed. All of them came from running the
thing, not from benchmarking it.

**Expert parallelism was missing.** We served 288 routed experts with `--tp-size 4` and no `--ep-size`,
while the SGLang cookbook recipe for this model uses TP4/**EP4**. Adding it moved every batch bucket:

| batch | TP4/EP1 (115k decode samples, 4.4 days) | **TP4/EP4 (39k samples, 13 h)** | delta |
|---|---|---|---|
| 1 | 291 tok/s · accept 3.16 | **412** · **4.19** | +42% |
| 2–4 | 608 · 2.80 | **701** · 2.85 | +15% |
| 5–8 | 941 · 2.79 | **1,133** · 2.81 | +20% |
| 9–16 | 1,275 · 2.79 | **1,665** · 2.89 | **+31%** |
| 17–32 | 1,783 · 2.82 | **1,999** · 2.90 | +12% |
| 33–48 | 2,185 · 2.73 | **2,712** · 2.80 | +24% |

**Read these with the caveats.** This is *production traffic*, not the controlled `bench/` runs that
produced the rest of this repo: the two columns are different days with different prompt mixes, and the
outer buckets are thin (83 samples at batch 1, 145 at 33–48). The two well-sampled buckets — 9–16 with
20,826 measurements and 17–32 with 12,487 — are the ones to trust, and they show +31% and +12%.
Acceptance also rose in every bucket, which we cannot explain: EP does not change the drafter's maths.
It may be a second-order effect of the lower graph ceiling changing which adaptive configuration gets
picked, or simply different content. We are reporting it, not claiming it.

**Context doubled, because production was rejecting work.** 262144 was turning away real requests at
373k–390k tokens. 524288 fixed that (**0 rejections in 13 h**), and the KV pool still holds 6.3 M tokens.

**And both had to be paid for.** EP4 plus the longer context left 176 MiB free per card and started
throwing request-level OOMs. `MEMF 0.75` bought the headroom back. Prefix-cache hit rate came back to
**95.3%** (it was 95.5% before), so the 9% smaller pool cost essentially nothing. Full story in
incidents 8, 9 and 10.

## Quick start

```bash
# machine launched from lmsysorg/sglang:glm-5.3-flash
git clone https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-b200 && cd glm-5.3-flash-sglang-4x-b200
hf download RadixArk/GLM-5.3-Flash-NVFP4 --local-dir /root/model-glm53-radix     # 190 GB
API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))') MODEL_PATH=/root/model-glm53-radix bash serve/serve.sh
```

Boot is ~5 min (weights from NVMe, then decode CUDA-graph capture up to batch 128). Warm the server
before measuring: the first request of each new shape pays FlashInfer/TRT-LLM MoE autotune and
DeepGEMM JIT (a 22 s prose run takes 34 s the first time; 32 streams run at 443 tok/s before warm-up
and 1,990 after).

## Measured (2026-09-11, idle server, `bench/` scripts, second run)

| axis | 4× RTX PRO 6000 (production) | 2× B200 TP2 | **4× B200 TP4** |
|---|---|---|---|
| pt-BR prose, 1 request (`prose_speed.py`) | ~100 tok/s | 215 | **218–228** |
| code, 1 request (`code_speed.py`; server log at batch 1) | 167–197 | 261–284 | 222 (log 271–291; real traffic today shows up to 320 at acceptance 3.9) |
| 8 streams × 400 tokens (`concurrency.py 8 400`) | 472–508 | 1,117 | **1,195** |
| 16 streams | — | 1,773 | 1,760 |
| 24 streams | 542 (its ceiling) | 2,420 | 2,377 |
| 48 streams | — | — | **3,890** |
| 64 streams | — | — | **4,850–4,910** (76 tok/s per stream, acceptance 2.38) |
| 96 streams | — | — | **6,030–6,060** (63 per stream, the sweet spot) |
| 128 streams | — | — | 5,370–5,420 (42 per stream; ~122 in flight, queue ≤ 6) |
| warm prefill, 27k-token prompt | 8.1–8.4k tok/s | 22–26k | **29k tok/s** (0.92 s) |
| KV pool (fp8) | 1.85 M tokens | 4.9 M | 7.2–9.8 M |

Details, including the stock-adaptive vs per-bucket comparison at 32–64 streams, in
[`docs/results.md`](docs/results.md). Functional gates (`bench/gates.py`): API, tool calling, thinking
on/off, vision (three images), prefix-cache report and 8-stream concurrency pass; the three that
fail are the known false negatives described there.

## What to take from it

- **TP4 does not beat TP2 at low concurrency.** Both are bound by per-step latency (~30 ms per decode
  step with speculation); splitting the same 18B-active forward over four cards saves no time. Use
  four B200 for ≥ 48 streams, or run two TP2 replicas behind a router for ≈ 2× the low-concurrency
  aggregate.
- **Pin the speculation policy per batch bucket.** The stock adaptive policy measures acceptance and
  drops the draft at high batch (acceptance 1.0 at batch 64 → 2.6–3.2k tok/s). Holding 2 draft steps
  from batch 40 up keeps acceptance at 2.4 and gives 4.9k. Same lesson as on the SM120 box.
- **96 streams is the ceiling, not 128.** 128 fits (no OOM at 169–181 GB per card) but aggregate
  throughput drops because the prefill of incoming requests competes with decode.
- **Use the cookbook checkpoint.** `nvidia/GLM-5.3-Flash-NVFP4` does not load on this image (see
  incidents); `RadixArk/GLM-5.3-Flash-NVFP4` loads with no patches and its chat template has vision.
- `--enable-cache-report` is what makes `usage.prompt_tokens_details.cached_tokens` non-zero.

## Pitfalls we hit (so you don't)

- **`nvidia/GLM-5.3-Flash-NVFP4` fails at load**: `model.layers.11.self_attn.kv_b_proj.weight` arrives
  BF16 `[32768, 512]` while the loader created an FP4 parameter `[8192, 256]`; the checkpoint's
  exclusion list is written with the `model.language_model.` prefix and does not match. Use RadixArk.
- **Do not apply the 0xSero loader patches here**: on build `fe236ea6c3` they fail at import
  (`is_blackwell_supported` no longer lives in `fp8_utils`). They are for the older build.
- **Adaptive spec format**: `{"1": {"candidate_steps": [5], "up_hysteresis": 0.0, "down_hysteresis":
  0.0, "ceiling_coeff": 0}, ...}`. The short form `{"1": [5]}` crashes the scheduler.
- **Default `reasoning_effort` is `max`** on this template: 6–8k reasoning tokens on an essay prompt.
  Send `reasoning_effort: high` (or `low`) from the client.
- **Vast.ai team instances and SSH**: the key must be registered on the *owner* account and, in our
  case, also written by the on-start command (`authorized_keys2` + `AuthorizedKeysFile` in
  `sshd_config.d`); the API's attach-ssh and the console button were not enough.

## Layout

```
serve/serve.sh            launch script (all flags above, env-overridable)
serve/adaptive_spec.json  per-bucket speculation config (1→5, ≥2→3, ≥40→2 steps)
serve/env.example         API_KEY / MODEL_PATH / CONC
bench/                    prose, code, concurrency, TTFT, gates, per-batch log parser (see bench/README.md)
docs/results.md           every number, per run
docs/incidents.md         what broke, in order, and the fix
```

MIT. Measurements by CulturaBuilder; the SGLang cookbook page for GLM-5.3-Flash is the upstream
reference for the base flags.
