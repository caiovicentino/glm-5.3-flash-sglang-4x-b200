#!/usr/bin/env bash
# GLM-5.3-Flash-NVFP4 (RadixArk) on 4x NVIDIA B200 (SM100) with SGLang.
# The official SGLang cookbook recipe for GLM-5.3-Flash on Blackwell (nvfp4 weights, fp8 KV, trtllm DSA
# kernels, EAGLE/MTP speculation) plus: adaptive speculation with one candidate per batch bucket,
# multimodal, reasoning/tool parsers, metrics, cache report and 8192-token prefill chunks.
# Usage: API_KEY=... MODEL_PATH=/root/model-glm53-radix bash serve/serve.sh
# Env knobs: TP (4) CONC (128) CONTEXT_LENGTH (262144) CHUNK (8192) MEMF (0.85) DSA (trtllm) KVDT (fp8_e4m3)
#            MOE (flashinfer_cutlass) ADAPTIVE_SPEC (./adaptive_spec.json) SPEC_ARGS EXTRA_ARGS PORT (8000)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/model-glm53-radix}"
KEY="${API_KEY:-$(cat /root/.api_key 2>/dev/null || true)}"
[ -n "$KEY" ] || { echo "API_KEY not set (export API_KEY=... or write /root/.api_key)"; exit 1; }
CONC="${CONC:-60}"          # CUDA-graph ceiling; see incident 9

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash}" \
  --quantization modelopt_fp4 \
  --tp-size "${TP:-4}" --ep-size "${EP:-4}" \
  --context-length "${CONTEXT_LENGTH:-524288}" \
  --dsa-prefill-backend "${DSA:-trtllm}" --dsa-decode-backend "${DSA:-trtllm}" \
  --kv-cache-dtype "${KVDT:-fp8_e4m3}" \
  --moe-runner-backend "${MOE:-flashinfer_cutlass}" \
  ${SPEC_ARGS---speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --speculative-adaptive} \
  --speculative-adaptive-config "${ADAPTIVE_SPEC:-$HERE/adaptive_spec.json}" \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --mem-fraction-static "${MEMF:-0.75}" \
  --cuda-graph-max-bs-decode "$CONC" --max-running-requests "${MAXREQ:-96}" \
  --chunked-prefill-size "${CHUNK:-4096}" --max-prefill-tokens "${CHUNK:-4096}" \
  --enable-cache-report --enable-multimodal --enable-metrics --media-url-max-file-size-mb 1024 \
  ${EXTRA_ARGS:-} --api-key "$KEY" --host 0.0.0.0 --port "${PORT:-8000}"
