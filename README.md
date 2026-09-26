# GLM-5.3-Flash-NVFP4 on 4× NVIDIA B200 with SGLang

Production recipe and field notes for serving **GLM-5.3-Flash-NVFP4** (320B total / 18B active; DSA sparse
attention + KDA linear attention; vision; MTP speculative decoding) on **four B200** with SGLang, for a real
community workload: **80–124k requests/day of agentic coding traffic, 88k-token average prompts, 95.6% of prompt
tokens served from the prefix cache.**

Almost every number here comes from production logs rather than synthetic benchmarks, and the repo keeps the
changes we rolled back next to the ones we kept.

> **Status — 2026-09-26**
> - **Production: v3.1** since 2026-09-26 04:03 UTC — TP4/EP4, 524k context, decode CUDA graphs to 96, MTP capped
>   at 3 steps, 700 KDA state slots, 4 decode rounds between prefill chunks, image pre-processing on the CPU.
>   Deployed by `ops/`-style swap with automatic rollback: 2 min 12 s down, 12/12 smoke tests.
> - **First 9 hours vs v2 the same week:** long decode stalls (≥ 5 s with ≥ 5 users waiting) **0/day vs 463/day**;
>   image pre-processing OOMs **0 vs 172** — the HTTP process no longer touches the GPU at all; per-user decode
>   speed unchanged ([`docs/results.md`](docs/results.md), phase 8).
> - History: v2 (2026-09-19 → 26) is the fallback profile; v3 ran 9 hours on 2026-09-23 and was rolled back for
>   the image OOM that v3.1 fixes ([incident 16](docs/incidents.md)).

Series: [4× RTX PRO 6000](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-rtx-pro-6000) (SM120, PCIe) ·
[2× B200](https://github.com/caiovicentino/glm-5.3-flash-sglang-2x-b200) (TP2) · this repo (4× B200, TP4/EP4).

## Quick start

```bash
# on a machine launched from lmsysorg/sglang:glm-5.3-flash
git clone https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-b200 && cd glm-5.3-flash-sglang-4x-b200
hf download RadixArk/GLM-5.3-Flash-NVFP4 --local-dir /root/model-glm53-radix        # 190 GB
export API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')
bash serve/serve.sh                      # production profile (v3.1)
# PROFILE=v2 bash serve/serve.sh         # the previous one, kept as fallback
python ops/smoke.py                      # 9 mandatory checks once it answers
```

Boot takes ~2–3 minutes from local NVMe (weights, then CUDA-graph capture). Warm the server before measuring:
the first request of each new shape pays MoE autotune and JIT. **To change a running production server, use
[`ops/swap.sh`](ops/)** — it checks the boot numbers and the smoke tests and rolls back by itself.

## Recipe card (v3.1, production)

| | |
|---|---|
| hardware | 4× B200 (183 GB, NVLink NV18 all-to-all), Xeon Platinum 8559C, 2 TB RAM (1.35 TiB usable in the container), NVMe — a Vast.ai datacenter offer at ~US$ 28/h |
| image | `lmsysorg/sglang:glm-5.3-flash` — SGLang `0.0.0.dev1+gfe236ea6c3.mmfix1`, flashinfer 0.6.17, torch 2.13 + cu130, driver 595. Nothing else to install |
| checkpoint | [`RadixArk/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4) (190 GB, the cookbook's): routed experts + MLPs NVFP4, attention / router / embeddings / lm_head / vision / MTP layer BF16. `nvidia/GLM-5.3-Flash-NVFP4` does not load on this image (incident 1) |
| parallelism | `--tp-size 4 --ep-size 4` (EP4 since 2026-09-18: +12% to +31% in the well-sampled batch buckets) |
| attention | `--dsa-prefill-backend trtllm --dsa-decode-backend trtllm --kv-cache-dtype fp8_e4m3` |
| MoE | `--moe-runner-backend flashinfer_cutlass --quantization modelopt_fp4` |
| speculation | EAGLE/MTP 3 steps, top-k 1, 4 draft tokens, `--speculative-adaptive` with [`serve/adaptive_spec_v3.json`](serve/adaptive_spec_v3.json): 3 steps at batch 1–39, 2 at batch ≥ 40. Production acceptance 2.8–2.9. Capping at 3 (v2 used 5 at batch 1) shrinks the KDA verify-snapshot buffer by 6.6 GB/GPU |
| memory | `--mem-fraction-static 0.75`, `--max-mamba-cache-size 700` → KV pool **6.39 M tokens**, 700 KDA state slots, **37.7 GB free per card after graph capture**. Do not raise MEMF with a long context: the DSA indexer buffer (`chunk × context × 4 B` = 8 GiB here) comes out of free memory (incident 11). Full breakdown in [`docs/memory-budget.md`](docs/memory-budget.md) |
| prefill | `--chunked-prefill-size 4096 --max-prefill-tokens 4096 --prefill-decode-interval 4` (~18–21k tok/s cold; 4 decode rounds run between chunks, so running requests keep streaming during a cold prefill) |
| concurrency | `--max-running-requests 96 --cuda-graph-max-bs-decode 96` — every batch size up to the cap has a graph (v2 stopped at 60 and fell to 13 tok/s per user above it, incident 14) |
| images | `--image-processor-backend pil`: pre-processing on the CPU, 5–63 ms per image; the HTTP process holds no GPU memory (incident 16) |
| context | `--context-length 524288` — raised from 262k because production was rejecting real 373k–390k-token requests |
| API | OpenAI-compatible on :8000, `--reasoning-parser glm45 --tool-call-parser glm47 --enable-multimodal --enable-cache-report --enable-metrics` |

## Profiles

`serve/serve.sh` reads a profile from [`serve/profiles/`](serve/profiles) with `PROFILE=…`:

| | v2 — previous, fallback | v3 — rolled back | **v3.1** — production |
|---|---|---|---|
| decode CUDA graphs up to | 60 | 96 | 96 |
| max running requests | 96 | 96 | 96 |
| MTP steps by batch | 5 @1 · 3 @2–39 · 2 @≥40 | 3 @1–39 · 2 @≥40 | same as v3 |
| KDA state slots | 519 (derived) | 700 | 700 |
| decode rounds between prefill chunks | 0 | 4 | 4 |
| image pre-processing | `cuda:0`, HTTP process | `cuda:0`, HTTP process | **CPU (PIL)** |
| KV pool | 6.33 M tokens | 6.39 M | 6.39 M |
| KDA verify snapshots per GPU | 19.89 GB | 13.26 GB | 13.26 GB |
| free per card after graph capture | 36.70 GB | 37.74 GB | ≈ v3 |

v3 gets graphs to 96 *and* more KDA slots at no cost in KV pool. The trick is capping MTP at 3 steps: the KDA
verify-snapshot buffer is sized by `(max running + 1) × max draft tokens × 34.2 MB`, and the 5-step candidate —
used only at batch 1, 3.6% of the time — was making every request slot pay for 6 snapshots.

## What we learned from 99 hours of production (v2)

Full numbers: [`docs/results.md`](docs/results.md), phase 6.

- **The box is latency-bound, not throughput-bound.** A queue formed in 18 of 371,931 decode samples.
  Concurrency p50 8, p99 35. At batch 27 each card draws ~600 of 1,000 W with the memory controller 26% busy.
  Step time barely grows between batch 36 and 53, so aggregate throughput scales almost linearly to the ~96 knee.
- **Per-user speed:** 306 tok/s at batch 1, 213 at 2–4, 168 at 5–8, 132 at 9–16, 102 at 17–32.
- **Cold prefills stalled decode for everyone** — 207 min/day, 817 stalls/day of ≥ 5 s with ≥ 5 users waiting,
  worst 25 s — because SGLang runs prefill chunks ahead of decode. `--prefill-decode-interval 4` (v3) turned
  those stalls into slowdowns: **0 in nine hours**. `--enable-mixed-chunk`, which batches prefill chunks
  *with* decode, is also accepted with EAGLE on this build — untested so far (see v3.1 below).
- **A TTFT floor of ~0.5 s lives in the single Python front-end**, growing to ~1–1.8 s with 34 running. The fix
  (`--tokenizer-worker-num`) is incompatible with `--api-key` (incident 15).
- **Both cache pools are full.** KV: 15 k of 6.33 M tokens free. KDA state: 3 of 519 slots free. A session is
  reusable only while both its KV and a KDA checkpoint survive, and ~1,000 requests/day compute ≥ 100k cold
  tokens. The structural fix — HiCache in host RAM — is blocked upstream for KDA models
  ([sglang#33713](https://github.com/sgl-project/sglang/issues/33713)).
- **The fastest lever is not in the server.** The chat template defaults to `reasoning_effort: max` (800–1,000
  reasoning tokens even on trivial turns), and agent harnesses that resend old reasoning degenerate over long
  sessions — see Pitfalls.

## Memory, in one paragraph

Per GPU: 44.8 GB of weights, 3.8 GB of MTP layer, **~39 GB of KDA state** (of which 19.9 GB are verify
snapshots in v2), ~46 GB of KV (target + draft), then CUDA graphs and ~37 GB free for dynamic buffers.
`--mem-fraction-static` moves memory between the static pools and free memory; `--max-running-requests`,
the MTP draft length and `--max-mamba-cache-size` are paid from the static budget, i.e. from the KV pool; CUDA
graphs and the DSA indexer buffer are paid from free memory. An earlier version of this README had the graph
part backwards — details and the correction in [`docs/memory-budget.md`](docs/memory-budget.md).

## v3.1 — how it was deployed, and how to redo it

```bash
NEW=v3.1 OLD=v2 MAMBA_SLOTS_EXPECT=700 SMOKE_IMAGE_CPU=1 \
SMOKE_EXPECT="prefill_decode_interval=4,cuda_graph_max_bs_decode=96,max_mamba_cache_size=700,image_processor_backend=pil" \
  setsid nohup bash ops/swap.sh >/dev/null 2>&1 </dev/null &
```

On 2026-09-26 it ran as a scheduled job at 04:00 UTC, with a rollback chain of v2 + `pil` → v2 → the original
recipe. It came up in 2 min 12 s and passed all 12 smoke tests; the HTTP process went from 720 MiB on GPU 0 in v2
to none at all. Still unmeasured after 9 hours: batch 1 (never occurred) and batches above 60 (peak was 57).
Image pre-processing on the CPU costs 5–63 ms per image; agent sessions that resend many screenshots every turn
will feel that in TTFT.

**After v3.1: `--enable-mixed-chunk`.** Older SGLang builds refused mixed chunked prefill with speculative
decoding; this one allows it for EAGLE (`SpeculativeAlgorithm.supports_mixed_chunk`), and a dry parse with the v3.1
flags keeps `enable_mixed_chunk=True`. It would let decode continue *inside* the prefill batches instead of
between them. Not yet validated with KDA state and DSA — one change at a time, after v3.1 has a clean 24 h.

## Checked and ruled out

Several flags exist, parse and boot, and do nothing for this model. Details and evidence in
[`docs/ruled-out.md`](docs/ruled-out.md):

- `--enable-linear-replayssm-spec` would free the KDA verify snapshots but only activates for Qwen3-Next-style
  GDN and Kimi-Linear configs — inert for `Glm5NextForConditionalGeneration`.
- HiCache: blocked by sglang#33713 for KDA models. `--enable-int8-mamba-checkpoint` is the alternative, but it
  excludes HiCache and draws on free memory.
- DP attention: no session-affinity dispatch in this build, and MTP + DP attention is not validated on this model.
- Prefill CUDA graphs are disabled for KDA; FlashInfer allreduce fusion is not wired for this architecture.
- Beyond ~96 concurrent requests, per-user speed drops without any gain in total throughput.

## Idle-server benchmark (2026-09-11, `bench/` scripts, second run, TP4 before EP4)

| axis | 4× RTX PRO 6000 | 2× B200 TP2 | **4× B200 TP4** |
|---|---|---|---|
| pt-BR prose, 1 request | ~100 tok/s | 215 | **218–228** |
| code, 1 request | 167–197 | 261–284 | 222 (log 271–291) |
| 8 / 16 / 24 streams | 472–508 / — / 542 | 1,117 / 1,773 / 2,420 | 1,195 / 1,760 / 2,377 |
| 48 / 64 streams | — | — | 3,890 / **4,850–4,910** |
| 96 streams | — | — | **6,030–6,060** (63 per stream — the knee) |
| 128 streams | — | — | 5,370–5,420 (42 per stream) |
| warm prefill, 27k-token prompt | 8.1–8.4k tok/s | 22–26k | **29k** |

Up to 24 streams, four B200 in TP4 are no faster than two in TP2 (both per-step-latency bound); the four cards
pay off above ~48 streams. The per-bucket speculation config is what makes 64+ streams work: the stock adaptive
policy drops drafting at batch 64 and loses ~40%.

## EP4 (2026-09-18/19)

| batch | TP4/EP1 (115k samples, 4.4 days) | TP4/EP4 (39k samples, 13 h) | delta |
|---|---|---|---|
| 1 | 291 tok/s · accept 3.16 | 412 · 4.19 | +42% |
| 2–4 | 608 · 2.80 | 701 · 2.85 | +15% |
| 5–8 | 941 · 2.79 | 1,133 · 2.81 | +20% |
| **9–16** | 1,275 · 2.79 | **1,665** · 2.89 | **+31%** |
| **17–32** | 1,783 · 2.82 | **1,999** · 2.90 | **+12%** |
| 33–48 | 2,185 · 2.73 | 2,712 · 2.80 | +24% |

Production traffic on different days: trust the two well-sampled buckets (9–16: 20,826 samples; 17–32: 12,487).
EP disables shared-experts fusion (incident 13); it was still a clear win. The longer context and EP4 together
cost the memory headroom that `MEMF 0.75` bought back (incident 12).

## Pitfalls

- **Agent harnesses: send `clear_thinking: true`.** The template defaults to keeping previous reasoning in the
  context. In one 11-hour agent session, reasoning degenerated into `o o o o:` filler after ~750 clean turns
  and then fed on itself. With six degenerate reasoning blocks in the history, 6 of 12 generations degenerated
  without `clear_thinking` and 0 of 12 with it (it also shrinks long contexts).
- **Never `enable_thinking: false` to save tokens** — the plan leaks into `content`. Use `reasoning_effort: low` or
  `high` instead (1 and 16 reasoning tokens on a simple question, against ~800 at the default `max`).
- **`/get_server_info` and the `server_args=` line at startup print `--api-key` in plain text.** Filter both
  before sharing logs or dumps.
- **`--tokenizer-worker-num > 1` is incompatible with `--api-key`**, and a dry parse does not tell you.
- **The HTTP process pre-processes images on `cuda:0`** unless `--image-processor-backend pil`; keep headroom on
  GPU 0 or move it to the CPU.
- **Above `--cuda-graph-max-bs-decode`, requests are not refused, they run ~4.6× slower.** Keep
  `--max-running-requests` ≤ the graph ceiling unless you have measured the ungraphed speed.
- **Adaptive-spec format:** `{"1": {"candidate_steps": [5], "up_hysteresis": 0.0, "down_hysteresis": 0.0,
  "ceiling_coeff": 0}, …}`. The short form `{"1": [5]}` crashes the scheduler.
- `--enable-cache-report` is what makes `usage.prompt_tokens_details.cached_tokens` non-zero.
- Do not apply the 0xSero loader patches on this build; use the RadixArk checkpoint instead of NVIDIA's.
- Vast.ai team instances: see incident 9 for SSH.

## Layout

```
serve/serve.sh               launch script — knobs via env, PROFILE=v2|v3|v3.1
serve/profiles/*.env         the three profiles above
serve/adaptive_spec*.json    per-bucket speculation (v2: 1→5, 2→3, 40→2 · v3: 1→3, 40→2)
ops/swap.sh                  production swap with memory guard, smoke tests and automatic rollback
ops/smoke.py                 post-boot checks (models, reasoning, streaming, tools, vision, cache, images off GPU…)
tools/log_eval.py            per-bucket decode speed, acceptance, ms/step, queueing and prefill stalls from logs
tools/metrics_summary.py     TTFT / inter-token / queue percentiles and per-request sizes from /metrics
bench/                       idle-server benchmarks (prose, code, concurrency, TTFT, gates)
docs/results.md              every number, by phase
docs/incidents.md            what broke, in order, and the fix
docs/memory-budget.md        per-GPU memory, and what each knob costs
docs/ruled-out.md            what we checked and why it does not apply
```

MIT. Measurements by CulturaBuilder. The SGLang cookbook page for GLM-5.3-Flash is the upstream reference for
the base flags.
