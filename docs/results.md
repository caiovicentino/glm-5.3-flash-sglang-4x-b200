# Results — 4× B200

Phases 1–5: controlled runs and the first production days (2026-09-11 → 19). Phases 6–7: the production eval
of v2 and the v3 trial (2026-09-22 → 23), measured from real traffic with `tools/log_eval.py` and
`tools/metrics_summary.py`.

## Phases 1–5 — 4× B200 TP4, 2026-09-11 (14:00–18:30 UTC−3) → 2026-09-19

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

---

## Phase 5 — EP4 in production, 2026-09-18/19

Not a `bench/` run. These come from the server's own `Decode batch` log lines over real traffic,
bucketed by `#running-req`, extracted with the same parser on both logs. Config before: TP4, EP1,
ctx 262144, MEMF 0.80, graphs 96. After: TP4, **EP4**, ctx **524288**, MEMF **0.75**, graphs 60,
chunk 4096 on both.

| batch | TP4/EP1 · 115,343 samples · 4.4 days | TP4/EP4 · 39,411 samples · 13 h | delta |
|---|---|---|---|
| 1 | 291 tok/s (p90 454) · accept 3.16 | 412 (p90 562) · 4.19 | +42% |
| 2–4 | 608 (838) · 2.80 | 701 (1,004) · 2.85 | +15% |
| 5–8 | 941 (1,260) · 2.79 | 1,133 (1,566) · 2.81 | +20% |
| 9–16 | 1,275 (1,767) · 2.79 | 1,665 (2,380) · 2.89 | +31% |
| 17–32 | 1,783 (2,559) · 2.82 | 1,999 (3,014) · 2.90 | +12% |
| 33–48 | 2,185 (3,248) · 2.73 | 2,712 (4,454) · 2.80 | +24% |
| 49–64 | 2,696 (4,051) · 2.44 | — (no samples yet) | — |

Prefill, full chunks only (partial chunks report inflated numbers because cached tokens count as
computed): **~20,880 tok/s** before. Prefix-cache hit rate 95.5% before, **95.3%** after.

Health over the 13 h after the MEMF fix: **0 OOM, 0 context-length rejections, 0 retracted requests**,
queue empty, 10 GiB free per card steady.

Reading: the gain is real and shows in every bucket, but it is measured on production traffic across
different days, so treat the well-sampled middle of the table as the result and the edges as
directional. The honest summary is **+12% to +31% where the data is thick**, plus a context limit that
stopped rejecting real work.

## Phase 6 — production eval of v2, 2026-09-19 01:00 → 2026-09-23 04:21 UTC (99.4 h)

Read-only: server logs, `/metrics`, `nvidia-smi`, and a few probe requests. Reproduce with
`python3 tools/log_eval.py --since "2026-09-19 01:00" --tz -3 <logs>`.

**Traffic.** 80–124k requests/day, 8–10 B prompt tokens/day, 70–117 M generated tokens/day, agentic coding.
Per request (first 3.6 days, `/metrics`): prompt 88,483 tokens mean (p50 81k, p90 186k, p99 288k), of which
**3,918 computed** (p50 403, p99 101k) — **95.6% from the prefix cache**; generated 975 mean (p50 311, p99 9,746).
Concurrency p50 8, p90 21, p99 35, max 96. A queue existed in 18 of 371,931 decode samples.

**Decode per batch bucket.** The machine lives at batch 2–32 (93% of the time):

| batch | share of time | tok/s | per request | acceptance | ms per verify step |
|---|---|---|---|---|---|
| 1 | 3.6% | 306 | 306 | 3.39 | 11.1 |
| 2–4 | 15.9% | 636 | 213 | 2.83 | 13.6 |
| 5–8 | 22.7% | 1,061 | 168 | 2.81 | 17.0 |
| 9–16 | 34.4% | 1,569 | 132 | 2.81 | 21.6 |
| 17–32 | 20.4% | 2,277 | 102 | 2.85 | 28.3 |
| 33–48 | 2.6% | 2,715 | 76 | 2.71 | 36.0 |
| 49–64 | 0.0% | 3,203 | 61 | 2.34 | 38.8 |
| **65–96, no graph** | 0.4% | **964** | **13** | 2.24 | **174.3** |

**It is not bound by bandwidth or power.** At batch 27: ~600 W of 1,000 W per card, clocks at the 1,965 MHz
maximum, no throttling, SM busy 72%, **memory controller busy 26%**. Step time barely grows from batch 36 to 53
(36 → 39 ms), so aggregate throughput scales almost linearly up to the knee (~96).

**Latency (first 3.6 days).** TTFT p50 0.71 s, p90 2.38 s, p99 9.98 s. Inter-token latency p50 4.5 ms, p99 135 ms.
Queue time p99 8.5 s.

**Where it goes:**

- **Cold prefills stall decode for everyone.** With `--prefill-decode-interval 0`, while one request prefills
  (0.194 s per 4,096-token chunk ≈ 21k tok/s), nobody else decodes and nobody new is admitted. 4,962 bursts/day of
  ≥ 3 chunks; stall p50 1.2 s, p90 7.2 s, p99 13.2 s, max 25 s; **207 min/day of stalled decode, 817 stalls/day of
  ≥ 5 s with ≥ 5 users waiting**. ~1,000 requests/day recompute ≥ 100k cold tokens. The worst ones were one
  ~480k-token session coming back every ~19 minutes and being re-prefilled from scratch.
- **A TTFT floor in the Python front-end.** "hi" takes 0.525 s on a calm server; a 100k-token cached prompt 0.445 s
  (size does not matter). With 34 running, the HTTP headers of a trivial request arrive after ~1 s and
  `/health` takes 1.76 s; the HTTP/tokenizer process sits at 87% of one core. The fix
  (`--tokenizer-worker-num`) cannot be combined with `--api-key` (incident 15).
- **The graph cliff above 60** — last row of the table (incident 14).
- **Both cache pools full.** KV: 1.04 M tokens active, 5.27 M cached, **15 k free** of 6.33 M. KDA state: 112
  slots active, 404 cached, **3 free** of 519. A session is reusable only while both its KV and a KDA checkpoint
  survive.

## Phase 7 — v3 in production, 2026-09-23 04:29 → 13:43 UTC

Profile `serve/profiles/v3.env`: graphs to 96, MTP capped at 3 steps, 700 KDA slots, 4 decode rounds between
prefill chunks. Boot: KV 6,389,760 tokens, KDA verify snapshots 13.26 GB (19.89 in v2), **37.74 GB free after
graph capture** (36.70 in v2). Nine mandatory smoke tests passed. First 8 hours (04:30 → 12:31, 36k requests):

| | v2, first 3.6 days | **v3, 8 h** | v2 again, 12.4 h after the rollback |
|---|---|---|---|
| window | 2026-09-19 → 22 | 2026-09-23 04:30 → 12:31 | 2026-09-23 13:46 → 09-24 02:08 |
| concurrency p50 | 8 | 14 | 19 |
| **stalls ≥ 5 s with ≥ 5 users decoding** | ~817/day | **0** | **497/day** |
| inter-token latency p50 / p99 | 4.5 / 135 ms | 7.2 / 54 ms | 7.5 / 58 ms |
| per request, batch 9–16 / 17–32 | 132 / 102 tok/s | 139 / 107 | 145 / 114 |
| TTFT p50 / p99 | 0.71 / 9.98 s | 0.74 / 9.09 s | 0.86 / 8.30 s |
| queue time p99 | 8.5 s | 7.0 s | 4.8 s |
| prefix-cache hit rate | 95.6% | 96.2% | 95.9% |

**What survives the like-for-like comparison is only the stall elimination.** The v2 run right after the rollback
saw similar traffic and still stalled 497 times a day; v3 stalled zero times. The p99 inter-token latency, the
per-request speed and TTFT move with the traffic mix, not with v3 — an earlier version of this page credited v3
with cutting p99 inter-token latency from 135 to 54 ms, and the third column shows that was the traffic.

Other caveats: batch 1 never occurred and the peak was exactly 60, so neither the 3-step cap at batch 1 nor
the 61–96 graphs were exercised. With the interleave on, `log_eval.py` shows prefill bursts capped at 10 chunks
— the decode log fires every 40 passes and 4 run between chunks — which is itself the evidence that decode keeps
running during a cold prefill.

At 12:20 UTC all four cards lost ~3.7 GB of free memory together; GPU 0 fell to 380 MiB and the image
pre-processor started failing (incident 16). Rolled back to v2 at 13:43 UTC.
