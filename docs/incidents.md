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
