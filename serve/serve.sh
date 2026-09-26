#!/usr/bin/env bash
# GLM-5.3-Flash-NVFP4 (RadixArk) on 4x NVIDIA B200 (SM100) with SGLang.
#
#   API_KEY=... bash serve/serve.sh                  # production profile (v3.1)
#   PROFILE=v2 API_KEY=... bash serve/serve.sh       # any profile in serve/profiles/; PROFILE=none = built-in defaults
#
# Profiles: v3.1 = production · v2 = previous, fallback · v3 = tried 2026-09-23 and rolled back.
# A profile sets the knobs below; to change one on top of a profile, copy the profile or use EXTRA_ARGS.
#
# Knobs: TP (4) EP (4) CONTEXT_LENGTH (524288) MEMF (0.75) CHUNK (4096) CONC (60, decode CUDA-graph ceiling)
#        MAXREQ (96) DSA (trtllm) KVDT (fp8_e4m3) MOE (flashinfer_cutlass) ADAPTIVE_SPEC (adaptive_spec.json)
#        SPEC_ARGS  MAMBA_SLOTS (--max-mamba-cache-size; unset = derived from MEMF)
#        PDI (--prefill-decode-interval; unset = 0)  IMAGE_PROC (--image-processor-backend auto|torchvision|pil)
#        EXTRA_ARGS  PORT (8000)  MODEL_PATH  SERVED_MODEL_NAME
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-v3.1}"
if [ "$PROFILE" != "none" ]; then
  P="$HERE/profiles/$PROFILE.env"
  [ -f "$P" ] || { echo "no such profile: $P" >&2; exit 1; }
  set -a; . "$P"; set +a
fi
MODEL_PATH="${MODEL_PATH:-/root/model-glm53-radix}"
KEY="${API_KEY:-$(cat /root/.api_key 2>/dev/null || true)}"
[ -n "$KEY" ] || { echo "API_KEY not set (export API_KEY=... or write /root/.api_key)" >&2; exit 1; }
SPEC_FILE="${ADAPTIVE_SPEC:-adaptive_spec.json}"
case "$SPEC_FILE" in /*) ;; *) SPEC_FILE="$HERE/$SPEC_FILE" ;; esac

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash}" \
  --quantization modelopt_fp4 \
  --tp-size "${TP:-4}" --ep-size "${EP:-4}" \
  --context-length "${CONTEXT_LENGTH:-524288}" \
  --dsa-prefill-backend "${DSA:-trtllm}" --dsa-decode-backend "${DSA:-trtllm}" \
  --kv-cache-dtype "${KVDT:-fp8_e4m3}" \
  --moe-runner-backend "${MOE:-flashinfer_cutlass}" \
  ${SPEC_ARGS---speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --speculative-adaptive} \
  --speculative-adaptive-config "$SPEC_FILE" \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --mem-fraction-static "${MEMF:-0.75}" \
  --cuda-graph-max-bs-decode "${CONC:-60}" --max-running-requests "${MAXREQ:-96}" \
  --chunked-prefill-size "${CHUNK:-4096}" --max-prefill-tokens "${CHUNK:-4096}" \
  ${MAMBA_SLOTS:+--max-mamba-cache-size "$MAMBA_SLOTS"} \
  ${PDI:+--prefill-decode-interval "$PDI"} \
  ${IMAGE_PROC:+--image-processor-backend "$IMAGE_PROC"} \
  --enable-cache-report --enable-multimodal --enable-metrics --media-url-max-file-size-mb 1024 \
  ${EXTRA_ARGS:-} --api-key "$KEY" --host 0.0.0.0 --port "${PORT:-8000}"
