# Incidents, in order

Everything that broke or misled us on this box, with the fix. Numbers 7 and 12 were corrected on 2026-09-23 —
the original text had the mechanism wrong; see [`memory-budget.md`](memory-budget.md).

## 2026-09-11 — bring-up

1. **`nvidia/GLM-5.3-Flash-NVFP4` does not load** on `lmsysorg/sglang:glm-5.3-flash` (`fe236ea6c3.mmfix1`):
   `model.layers.11.self_attn.kv_b_proj.weight` arrives BF16 `[32768, 512]` while the loader created an FP4
   parameter `[8192, 256]`. NVIDIA's exclusion list uses the `model.language_model.` prefix, which does not
   match SGLang's module names. Fix: the cookbook checkpoint, `RadixArk/GLM-5.3-Flash-NVFP4`.

2. **The 0xSero loader patches break the import** on this build (`ImportError: is_blackwell_supported` from
   `fp8_utils`). They were for the older build of the 2× B200 recipe; with the RadixArk checkpoint TP4 loads on
   the stock tree.

3. **Stock adaptive speculation collapses at 64 streams**: the policy drops the draft (acceptance 1.0) and the
   aggregate falls to 2.6–3.2k tok/s. Fix: one candidate per batch bucket (`serve/adaptive_spec.json`) →
   4.85–4.9k at 64 streams.

4. **The short adaptive-spec form `{"1": [5]}` crashes the scheduler** (`'list' object has no attribute 'get'`).
   Use the long form.

5. **The first pass of each shape is slow**: prose 34 s vs 22 s, 32 streams 443 vs 1,990 tok/s — MoE autotune
   and JIT. Warm the server before measuring or before putting users on it.

6. **128 streams fit but are slower than 96** (5.4k vs 6.0k aggregate): prefill of new arrivals competes with
   decode. The knee on this box is ~96.

7. **(corrected 2026-09-23) The pool shrank from 9.77 M to 7.19 M tokens between the 32- and 128-stream runs —
   because of `--max-running-requests`, not the graph ceiling.** `CONC` raised both. Each running-request slot
   reserves KDA verify snapshots (`draft_tokens × 34.2 MB` per GPU), taken from the static budget that the KV pool
   also uses; graphs are captured afterwards, from free memory.

8. **`usage.prompt_tokens_details.cached_tokens` stays 0 without `--enable-cache-report`**, even with the radix
   cache active.

9. **SSH on a Vast.ai team instance**: register the key on the team *owner* account and create the instance via
   the API with an on-start command that writes `authorized_keys2`, adds `AuthorizedKeysFile .ssh/authorized_keys
   .ssh/authorized_keys2` under `sshd_config.d` and restarts sshd. The console button and attach-ssh were not
   enough.

10. **`reasoning_effort` defaults to `max`** in the checkpoint's template (6–8k reasoning tokens on an essay
    prompt). Send `high` or `low` from the client when you do not need the maximum.

## 2026-09-14 → 19 — production

11. **`MEMF 0.85` + chunk 8192 + a long context killed the engine on all four ranks** (2026-09-14):
    `fp8_mqa_logits` asked for 9.25 GiB with 8.5 GiB free. The DSA indexer buffer is
    `chunked_prefill_size × context_length × 4 bytes` (9.25 GiB = 8192 × ~303k × 4), and it comes out of the memory
    `--mem-fraction-static` leaves free. Context and chunk compete for the same bytes: at 524k, chunk 4096 costs
    8 GiB and chunk 8192 costs 16 GiB. Defaults since: `MEMF 0.75`, chunk 4096.

12. **(corrected 2026-09-23) EP4 + 524k context left ~25 GB free after graph capture and produced 11
    request-level OOMs** (`Tried to allocate 214.00 MiB`) in 83 minutes; the server stayed up. Fix: `MEMF 0.80 →
    0.75` (+8.8 GB free), with graphs 96 → 60 adding ~2.8 GB more. The original text said lowering the graph
    ceiling "returns pool, not free memory" — the opposite is true: graphs come out of free memory, so they do
    help, just less than MEMF.

13. **EP disables shared-experts fusion** (`Shared experts fusion is not supported together with expert parallelism
    yet`). The model has one shared expert next to the 288 routed ones; EP4 still won by +12% to +31% per
    well-sampled batch bucket.

## 2026-09-22 → 23 — the v3 trial

14. **Above the CUDA-graph ceiling a batch is not refused — it runs 4.6× slower.** v2 sets `--max-running-requests
    96` with graphs to 60, on the assumption that the excess would otherwise be rejected. It is not: above the
    running cap requests wait in the queue. What 61–96 actually did in production was run without a graph at
    **174 ms per step, 13 tok/s per user**, against 37 ms and 63 tok/s at 49–64. Rare (0.4% of the time), but
    exactly at peaks. Fix: graphs to 96 (v3), or `--max-running-requests` equal to the graph ceiling.

15. **`--tokenizer-worker-num 4` with `--api-key` kills the boot** — after the weights load, at HTTP startup:
    `AssertionError: API key is not supported in multi-tokenizer mode`. A dry parse with `prepare_server_args`
    does not catch it. `ops/swap.sh` rolled production back to v2 automatically (4 min 37 s from stop to serving).
    With a public port the key cannot go, so the front-end TTFT floor stays until a proxy or upstream change.

16. **v3 at peak: the image pre-processor ran out of memory on GPU 0.** The HTTP/tokenizer process runs the fast
    image processor on `cuda:0` (`base_processor._fast_image_processor_device`). In v2, GPU 0 happens to plateau
    with ~28 GB free; in v3 memory settled evenly (~4.5 GB free per card), and at 12:20 UTC a synchronised ~3.7 GB
    step left GPU 0 with 380 MiB. Result: **56 `OutOfMemoryError`, all in
    `image_processing_glm5_next.py::_preprocess`** (a 298 MiB `torch.cat`) — only requests carrying images failed;
    inference itself logged a single allocator warning in nine hours. Rolled back to v2 at 13:43 UTC (2 min 16 s
    down). **Not a v3 problem:** back on v2, the same failure hit on 2026-09-24/25 once GPU 0 lost its accidental
    headroom — 172 image OOMs, up to ~56 an hour, all in `_preprocess`. Fix, in production since 2026-09-26 as
    v3.1: `--image-processor-backend pil`, which makes SGLang skip the device entirely; the PIL processor produces
    the same tensors (max abs diff < 5e-4) at 5–63 ms of CPU per image, and the HTTP process now holds no GPU
    memory at all.
