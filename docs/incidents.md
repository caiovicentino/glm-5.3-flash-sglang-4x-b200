# Incidents, in order (2026-09-11)

1. **`nvidia/GLM-5.3-Flash-NVFP4` does not load** on `lmsysorg/sglang:glm-5.3-flash` (build
   `fe236ea6c3.mmfix1`): `model.layers.11.self_attn.kv_b_proj.weight` comes as BF16 `[32768, 512]`
   and the loader had created an FP4 parameter `[8192, 256]`. NVIDIA's `exclude_modules` list is
   written with the `model.language_model.` prefix, which does not match the module names SGLang
   builds. Fix: the cookbook checkpoint, `RadixArk/GLM-5.3-Flash-NVFP4`, whose exclusion list carries
   both spellings.

2. **The 0xSero loader patches break the import** on this build (`ImportError: is_blackwell_supported`
   from `fp8_utils`). They were needed by the 2× B200 recipe on the older `f13cb6f6a7` build. Here,
   with the RadixArk checkpoint, TP4 loads on the stock tree. Revert the patches if you applied them.

3. **Stock adaptive speculation collapses at 64 streams**: acceptance reads 1.0 (the policy dropped
   the draft) and aggregate falls to 2.6–3.2k tok/s. Fix: `--speculative-adaptive-config` with one
   candidate per batch bucket (`serve/adaptive_spec.json`) → 4.85–4.9k at 64 streams.

4. **Adaptive spec written in the short form** `{"1": [5]}` crashes the scheduler
   (`'list' object has no attribute 'get'`). Use the long form in `serve/adaptive_spec.json`.

5. **First pass per shape is slow**: prose 34 s vs 22 s, 32 streams 443 vs 1,990 tok/s. MoE autotune
   and JIT, not a regression. Warm the server before measuring or before putting users on it.

6. **128 streams fits but is slower than 96** (5.4k vs 6.0k aggregate). Prefill of new arrivals competes
   with decode; the scheduler settles at ~122 running. Keep `CONC=128` for headroom, expect the
   optimum around 96.

7. **Raising the CUDA-graph ceiling costs KV pool**: 9.77 M tokens with graphs to 32, 7.19 M with
   graphs to 128. Still 4× the SM120 production pool.

8. **`usage.prompt_tokens_details.cached_tokens` stays 0** without `--enable-cache-report`, even with
   the radix cache active; with it, a repeated 8k system prompt reports 99 % cached.

9. **SSH on a Vast.ai team instance**: neither the console's SSH button nor the API's attach-ssh
   put our key on the box. What worked: register the key on the *owner* account of the team and
   create the instance through the API with an on-start command that writes `authorized_keys2`,
   adds `AuthorizedKeysFile .ssh/authorized_keys .ssh/authorized_keys2` under `sshd_config.d`, and
   restarts sshd.

10. **`reasoning_effort` defaults to `max`** in the checkpoint's template: 6–8k reasoning tokens on an
    essay prompt, coherent and without repetition, just long. Send `high` or `low` from the client.

---

## Added 2026-09-19, after 8 days of continuous production

8. **`MEMF 0.85` + `chunk 8192` is not safe once you raise the context.** On 2026-09-14 01:25 UTC the
   engine died on all four ranks: `fp8_mqa_logits` asked for **9.25 GiB** with **8.5 GiB free**. The DSA
   indexer buffer is **`chunked_prefill_size × context_length × 4 bytes`** — the 9.25 GiB reconstructs
   exactly as `8192 × ~303k × 4`. It is linear in *both* factors, and it comes out of the memory
   **`--mem-fraction-static` leaves free**, not out of the KV pool. So context length and prefill chunk
   compete for the same bytes: with `ctx 524288`, chunk 4096 costs 8 GiB and chunk 8192 costs 16 GiB.
   The README used to say "the checkpoint supports more; the pool has room" about context — that advice
   was wrong, and this is the correction. Defaults here are now `MEMF 0.75`, `chunk 4096`.

9. **EP4 is worth it, but it eats the memory headroom.** Adding `--ep-size 4` and doubling context to
   524288 (2026-09-18 23:16 UTC) left **176 MiB free per card** and produced **11 request-level
   `torch.OutOfMemoryError`** (`Tried to allocate 214.00 MiB`) over 83 minutes. The server did **not**
   die — the KV pool never filled and nothing was retracted; transient buffers simply had nowhere to go.
   Fix: `MEMF 0.80 → 0.75`, which restored ~21 GiB free and stopped the OOMs dead (**0 in the following
   13 hours**).

   The trap worth naming: **lowering the CUDA-graph ceiling does not fix this.** Graphs come out of the
   same static budget as the KV pool (incident 7), so cutting them returns *pool*, not *free memory*.
   Only `--mem-fraction-static` controls what stays free. We dropped graphs 96 → 60 anyway, for a
   different reason: to win back the pool that the lower MEMF costs. It covers the real traffic — the
   largest batch seen under EP4 is 41, and in 115k historical decode samples the 49–64 bucket is 0.3%
   with nothing above 64. Batches above 60 are still *accepted* (`--max-running-requests 96`); they just
   run without a captured graph.

10. **EP disables shared-experts fusion.** The log says so plainly:
    `Shared experts fusion is not supported together with expert parallelism yet`. The model has 1 shared
    expert alongside the 288 routed ones, so this is a real optimisation traded away. The throughput
    below says the trade is worth taking, but it is a trade, not a free win.
