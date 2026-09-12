# Results — 4× B200 TP4, 2026-09-11 (14:00–18:30 UTC−3) and 2026-09-12

Server idle between runs; each script run at least twice; numbers are the second (warm) run unless
stated. Checkpoint `RadixArk/GLM-5.3-Flash-NVFP4`, image `lmsysorg/sglang:glm-5.3-flash` (build
`fe236ea6c3.mmfix1`). 2× B200 column: sibling recipe, same scripts, same day. Production column:
4× RTX PRO 6000 SM120, measured 2026-09-04…09-10.

## Phase 1 — cookbook flags, CONC=32, stock adaptive speculation

| axis | 2× B200 TP2 | 4× B200 TP4 |
|---|---|---|
| prose, batch 1 (`prose_speed.py`) | 215 tok/s | 218 |
| code, batch 1 (`code_speed.py`) | 261–284 | 222 (server log at batch 1: 271–291, acceptance 3.0–3.3) |
| 8 streams | 1,117 | 1,014–1,084 |
| 16 streams | 1,773 | 1,760 |
| 24 streams | 2,420 | 2,377 |
| 32 streams | — | 1,940–1,990 (acceptance falls to 1.84, the adaptive policy shortens the draft) |
| warm prefill, 27k-token prompt | 22–26k tok/s | 29k tok/s (0.92 s) |
| KV pool | 4.9 M tokens | 9.77 M |

Reading: up to 24 streams TP4 on four cards yields nothing over TP2 on two. The limit is per-step
latency. The first pass of every shape pays MoE autotune (FlashInfer/TRT-LLM) and JIT: prose 34 s vs
22 s, 32 streams 443 vs 1,990 tok/s.

## Phase 2 — CONC=64, graphs up to 64: stock adaptive vs per-bucket

| streams | stock adaptive | per-bucket (`adaptive_spec.json`: 1→5, ≥2→3, ≥40→2 steps) |
|---|---|---|
| 1 | 218 tok/s | 228 |
| 8 | 1,014–1,084 | 1,195 |
| 32 | 2,300–2,760 | — |
| 48 | 3,600–3,700 | 3,890 |
| 56 | 4,120 | 4,210 |
| 64 | 2,600–3,160 (acceptance 1.0: drafting switched off) | **4,850–4,910** (76 tok/s per stream, acceptance 2.38) |

Same lesson as on the SM120 production box: the stock adaptive policy disables speculation at high
batch; one candidate per bucket holds 2 draft steps and gives +60 % at 64 streams without losing
anything at batch 1.

## Phase 3 — CONC=128, graphs up to 128, per-bucket speculation

| streams | aggregate (client) | per stream | decode in server log |
|---|---|---|---|
| 64 | 4,820 tok/s | 75 | 4,380 (acceptance 2.36) |
| 96 | 6,030–6,060 | 63 | 5,800 (2.39) |
| 128 | 5,370–5,420 | 42 | 6,190 at batch 122 (2.43); queue ≤ 6 |

128 fits (169 GB per card during the run, no OOM, no aborts) but the optimum is ~96: above it the
client-side aggregate stops growing because the prefill of new requests competes with decode and
~122 actually run. Throughput regime: 6k tok/s = 11× the production ceiling (542).

The pool shrinks with the graph ceiling: 9.77 M tokens with `--cuda-graph-max-bs-decode 32`,
7.19 M with 128 (`max_running_requests=122` effective, `available_gpu_mem=17.4 GB` after capture).

## Production traffic, 2026-09-12

The box served ~90k requests / 4.9 B prompt tokens / 61 M generated tokens in its first 23 h. Batch-1
decode in the log at that point: 320–330 tok/s with acceptance 3.8–3.9 (code-heavy traffic).

## Functional gates (`bench/gates.py`)

6/9 pass; the three failures are the known ones: B (long pt-BR essay) is a reasoning budget effect of
`reasoning_effort: max`, H3 (shapes image) is a strict string match rejecting a correct answer, and G
(cached tokens) passes once `--enable-cache-report` is on (99 % cached on a repeated 8k system prompt).
Tool calling, thinking on/off, the two other vision gates and 8-stream concurrency pass.
