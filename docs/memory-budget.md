# Memory budget per GPU — what each knob really costs

4× B200, GLM-5.3-Flash-NVFP4, TP4/EP4, SGLang `fe236ea6c3.mmfix1`. Numbers from the TP0 boot log of each
configuration; the other ranks match within ~0.2 GB. Each card reports 178.34 GiB; after CUDA and NCCL init,
176.25 GB are available to SGLang.

| | v1 (2026-09-18, OOMed) | **v2 (production)** | v3 (2026-09-23) |
|---|---|---|---|
| `--mem-fraction-static` | 0.80 | 0.75 | 0.75 |
| max running requests / MTP draft tokens | 96 / 6 | 96 / 6 | 96 / 4 |
| decode CUDA graphs up to | 96 | 60 | 96 |
| weights: target + MTP draft layer | 44.76 + 3.81 GB | same | same |
| KDA state slots (`max_mamba_cache_size`) | 574 (derived) | 519 (derived) | 700 (explicit) |
| KDA recurrent state + conv state | 19.65 + 0.69 GB | 17.77 + 0.62 GB | 23.96 + 0.84 GB |
| **KDA verify snapshots** (`intermediate_ssm_state_cache`) | 19.89 GB | **19.89 GB** | **13.26 GB** |
| KDA conv window for verify | 0.70 GB | 0.70 GB | 0.47 GB |
| KV pool, fp8 | 48.09 GB = 7,289,664 tokens | 41.75 GB = 6,327,680 | 42.16 GB = 6,389,760 |
| KV of the draft layer | 4.37 GB | 3.80 GB | 3.83 GB |
| free after the pools | 34.00 GB | 42.86 GB | 42.82 GB |
| CUDA graphs (captured after the pools) | ~9 GB | 6.16 GB | 5.08 GB |
| **free after graph capture** | **~25 GB → request-level OOMs** | **36.70 GB** | **37.74 GB** |

## The formulas behind it

- **KDA state:** 34.2 MB per slot per GPU (34 linear-attention layers, heads split across TP ranks).
- **KDA verify snapshots:** `(max_running_requests + 1) × speculative_num_draft_tokens × 34.2 MB`. This is the
  largest single item after the weights and the KV pool: 97 × 6 × 34.2 MB = 19.9 GB. It is sized by the
  *maximum* draft tokens across the adaptive candidates, so a 5-step candidate used only at batch 1 makes every
  request slot pay for 6 snapshots.
- **KV:** 6.6 KB per token **per GPU**. The DSA latent is not split across TP ranks — every card holds the whole
  KV of every cached token.
- **DSA indexer buffer** (dynamic): `chunked_prefill_size × context_length × 4 bytes` = 8 GiB at chunk 4096 and
  524k context. It comes out of free memory (incident 11).

## What each knob costs, and where from

| knob | costs | taken from |
|---|---|---|
| `--mem-fraction-static` | 8.8 GB per 0.05 | moves memory between the static pools and free memory |
| `--max-running-requests` | ~205 MB/GPU per request slot with 6 draft tokens (~137 MB with 4) | static budget → shrinks the KV pool |
| MTP draft tokens | same formula, times max running requests | static budget |
| `--max-mamba-cache-size` | 34.2 MB + conv per slot | static budget → shrinks the KV pool |
| `--cuda-graph-max-bs-decode` | ~1 GB per step-count set when going 60 → 96 | **free memory** |
| chunk × context | the indexer buffer above | **free memory** |

> **Correction.** Earlier versions of this repo said CUDA graphs "come out of the same static budget as the KV
> pool", and blamed the pool shrink of the 2026-09-11 runs (9.77 M tokens with graphs to 32, 7.19 M with graphs
> to 128) on the graph ceiling. Both were wrong. The v1/v2 boot logs show the pools differing by 8.86 GB — exactly
> the MEMF change (0.05 × 176.25 GB) — with graphs captured afterwards from free memory. What shrank the pool in those runs
> was `--max-running-requests`, which `CONC` raised together with the graph ceiling, through the verify
> snapshots.

## Steady state

The PyTorch caching allocator keeps most of the free memory within hours, so `nvidia-smi` "free" is not
headroom. What matters is the plateau and whether anything else needs memory on the same card:

- v2 plateau: **GPU 0 ~28 GB free, GPUs 1–3 ~0.5 GB** — an asymmetry we have not explained.
- v3 plateau: ~4.5 GB on all four cards for eight hours, then ~0.4–3 GB after a synchronised ~3.7 GB step at
  peak load.
- The HTTP/tokenizer process pre-processes images on `cuda:0` unless `--image-processor-backend pil` is set. In
  v2 it lived off GPU 0's accidental 28 GB; in v3 it ran out (incident 16).
