#!/usr/bin/env bash
# Safe production swap: start a new profile, check the boot numbers and the smoke tests, and roll back to the
# old profile automatically if anything fails. Run it DETACHED so an SSH drop cannot cut it in half:
#
#   NEW=v3.1 OLD=v2 SMOKE_IMAGE_CPU=1 setsid nohup bash ops/swap.sh >/dev/null 2>&1 </dev/null &
#   tail -f /root/swap.log            # ends with END_OK or END_ROLLED_BACK
#
# Env: NEW, OLD             profile names from serve/profiles/
#      SERVER_LOG           /root/glm.log   (the server's stdout/stderr)
#      SWAP_LOG             /root/swap.log
#      MIN_FREE_GB          35.5   free memory on TP0 after CUDA-graph capture (v2 36.70, v3 37.74; v1 OOMed ~25)
#      MIN_KV_TOKENS        5800000
#      MAMBA_SLOTS_EXPECT   exact KDA slot count to require (optional, e.g. 700)
#      BOOT_TIMEOUT         900 s
#      PY                   python with PIL + numpy for the smoke tests (default: python)
#      SMOKE_*              passed through to ops/smoke.py
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW="${NEW:?set NEW=<profile>}"; OLD="${OLD:?set OLD=<profile>}"
SERVER_LOG="${SERVER_LOG:-/root/glm.log}"; L="${SWAP_LOG:-/root/swap.log}"
KEY="${API_KEY:-$(cat /root/.api_key 2>/dev/null || true)}"; export API_KEY="$KEY"
say()        { echo "[$(date -u +%H:%M:%S)] $*" >> "$L"; }
code()       { curl -s -m 10 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" "http://127.0.0.1:${PORT:-8000}/v1/models"; }
stop()       { for p in $(pgrep -f "[l]aunch_server"); do kill "$p" 2>/dev/null; done; sleep 8
               for p in $(pgrep -f "[l]aunch_server"); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
start()      { [ -s "$SERVER_LOG" ] && mv "$SERVER_LOG" "$SERVER_LOG.pre-$1-$(date -u +%Y%m%d-%H%M%S)"
               : > "$SERVER_LOG"; PROFILE="$1" setsid nohup bash "$HERE/serve/serve.sh" >> "$SERVER_LOG" 2>&1 < /dev/null & }
wait_ready() { local t=0; while [ "$t" -lt "${BOOT_TIMEOUT:-900}" ]; do sleep 5; t=$((t+5))
               [ "$(code)" = "200" ] && return 0; pgrep -f "[l]aunch_server" >/dev/null || return 2; done; return 1; }
rollback()   { say "ROLLING BACK to $OLD — $1"
               cp "$SERVER_LOG" "$SERVER_LOG.failed-$NEW-$(date -u +%Y%m%d-%H%M%S)" 2>/dev/null
               stop; start "$OLD"
               if wait_ready; then say "$OLD back up"; else say "!!! $OLD did not come up — MANUAL INTERVENTION"; fi
               say "END_ROLLED_BACK"; exit 1; }

say "START swap $OLD -> $NEW"
t0=$(date +%s); stop; start "$NEW"
wait_ready; rc=$?
[ "$rc" -eq 0 ] || rollback "$NEW not ready (rc=$rc: 1=timeout, 2=process died)"
say "$NEW ready in $(( $(date +%s) - t0 )) s"

SLOTS=$(grep -a -m1 "Mamba Cache is allocated" "$SERVER_LOG" | sed -E 's/.*max_mamba_cache_size: ([0-9]+).*/\1/')
KV=$(grep -a -m1 "KV Cache is allocated" "$SERVER_LOG" | sed -E 's/.*#tokens: ([0-9]+).*/\1/')
FREE=$(grep -a "TP0" "$SERVER_LOG" | grep -a "Capture .* end" | sed -E 's/.*avail mem=([0-9.]+) GB.*/\1/' | sort -n | head -1)
say "boot: kda_slots=$SLOTS kv_tokens=$KV free_after_graphs=${FREE}GB"
awk -v s="$SLOTS" -v k="$KV" -v f="$FREE" -v ms="${MAMBA_SLOTS_EXPECT:-}" \
    -v mk="${MIN_KV_TOKENS:-5800000}" -v mf="${MIN_FREE_GB:-35.5}" \
    'BEGIN { exit !(k >= mk && f >= mf && (ms == "" || s == ms)) }' || rollback "memory guard failed"
say "memory guard OK — running smoke tests"
"${PY:-python}" "$HERE/ops/smoke.py" >> "$L" 2>&1 || rollback "a mandatory smoke test failed"
say "$NEW LIVE AND VALIDATED in $(( $(date +%s) - t0 )) s"
say "END_OK"
