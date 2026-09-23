# What we checked and ruled out

Build `lmsysorg/sglang:glm-5.3-flash` (`fe236ea6c3.mmfix1`), 2026-09-22/23. Every verdict comes from reading
the build's source or from measuring on the box — not from flag names. Several of these flags exist, parse and
boot fine, and do nothing for this model — and one thing we had written off turned out to be allowed on this
build.

| idea | verdict | evidence |
|---|---|---|
| `--enable-linear-replayssm-spec` — would drop the 13–20 GB/GPU of KDA verify snapshots | **inert on GLM-5.3** | The pool code activates it only when `hybrid_gdn_config` (Qwen3-Next family) or `kimi_linear_config` matches. For `Glm5NextForConditionalGeneration` both return `None` (checked by calling them). |
| HiCache — host-RAM tier (the container may use 1.35 TiB) | **blocked by [sglang#33713](https://github.com/sgl-project/sglang/issues/33713)** | With a MAMBA/KDA component on `UnifiedRadixCache`, device eviction *prunes* the node instead of downgrading it to host; the host copy is written and never loaded back. Open since 2026-08-05, reproduced independently on v0.5.19. The cookbook's "no regression with HiCache" run used random prompts with no prefix reuse, so it never hit this path. Useful if it gets fixed: `--hicache-size` is per TP rank and is split between the KV and KDA host pools in proportion to their device sizes. |
| `--enable-int8-mamba-checkpoint` — 2× cached KDA states | possible, not tried | Model-agnostic, but mutually exclusive with HiCache, and its int8 pool is sized from free memory — the headroom that request-level OOMs come from. |
| DP attention — would quadruple KV capacity (the DSA latent is replicated across TP ranks) | **wrong for multi-turn traffic** | `--load-balance-method` offers `round_robin`, `total_requests`, `total_tokens` (+ PD modes): no session affinity, so turns would land on ranks without their prefix and a 95.6% hit rate is what keeps this box in the shallow regime. The official cookbook also states that MTP + DP attention is not validated on this model. |
| `--tokenizer-worker-num > 1` — to lift the single-process front-end's TTFT floor (0.5 s idle, `/health` 1.76 s at 34 running) | **incompatible with `--api-key`** | `AssertionError: API key is not supported in multi-tokenizer mode` from `http_server.init_multi_tokenizer`, after ~2 minutes of model loading. `prepare_server_args` does not catch it (incident 15). |
| Prefill CUDA graphs | disabled by the build | "Breakable CUDA graph is incompatible with KDA hybrid linear attention"; the cookbook's breakable prefill graphs need PR #38522. Every small prefill runs eager (~0.18 s median `prefill_forward`). |
| FlashInfer allreduce + RMSNorm fusion on NVLink | not wired for this architecture | The auto-enable list has `GlmMoeDsaForCausalLM` (GLM-5/5.2), not `Glm5NextForConditionalGeneration`. |
| `--enable-mixed-chunk` (vLLM-style mixed prefill + decode batches) | **not ruled out — untested** | Older builds refused it with speculative decoding; this one allows it for EAGLE (`SpeculativeAlgorithm.supports_mixed_chunk` lists EAGLE, EAGLE3, DFLASH, DSPARK), and it survives argument resolution with our flags. The next thing to try after v3.1; v3 used `--prefill-decode-interval 4` instead. |
| Going past ~96 concurrent | past the knee | 2026-09-11: 96 streams 6,030–6,060 tok/s (63 per stream); 128 streams 5,370–5,420 (42 per stream). |
| Fixed-depth MTP at high batch (cookbook campaign: 8,490 vs 3,665 tok/s at c80) | does not transfer | Measured with random prompts at temperature 0 (acceptance 4.7–6.0) against the stock adaptive table, which turns drafting off from batch 64. Our production acceptance is 2.3–2.9, and the per-bucket config already keeps 2 steps from batch 40 up (1.6–1.9× over drafting off at 64 streams). |
| Profilers (py-spy, CUPTI) | unavailable on Vast.ai containers | No ptrace permission. Attribution here comes from logs, `/metrics` and `nvidia-smi`. |
| `sglang:evicted_tokens_total` as cache turnover | misleading | It reported ~4.8 B tokens evicted over 3.6 days against ~1.7 B that could have entered the cache. Use the pool gauges (`kv_available_tokens`, `mamba_available_tokens`) instead. |
