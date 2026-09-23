#!/usr/bin/env python3
"""Latency percentiles and per-request sizes from an SGLang /metrics endpoint (cumulative since server start).

    python3 tools/metrics_summary.py [http://127.0.0.1:8000/metrics | saved_metrics.txt]

Percentiles are interpolated inside Prometheus buckets. Some histograms are exported once per TP rank (their
counts are 4x the request count on TP4); percentiles are unaffected. Do not read sglang:evicted_tokens_total as
KV turnover: on our server it reported ~2.8x more tokens evicted than could have entered the cache.
"""
import collections, re, sys, urllib.request

src = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/metrics"
text = urllib.request.urlopen(src, timeout=20).read().decode() if src.startswith("http") else open(src).read()
hist = collections.defaultdict(lambda: collections.defaultdict(float)); sums = collections.defaultdict(float)
counts = collections.defaultdict(float); scal = collections.defaultdict(list)
for line in text.splitlines():
    m = re.match(r'^([a-zA-Z_:]+)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$', line)
    if not m or line.startswith("#"):
        continue
    name, lab, val = m.group(1), m.group(2) or "", float(m.group(3))
    st = re.search(r'stage="([^"]+)"', lab); suffix = f"[{st.group(1)}]" if st else ""
    if name.endswith("_bucket"):
        le = re.search(r'le="([^"]+)"', lab).group(1)
        hist[name[:-7] + suffix][float("inf") if le == "+Inf" else float(le)] += val
    elif name.endswith("_sum"):
        sums[name[:-4] + suffix] += val
    elif name.endswith("_count"):
        counts[name[:-6] + suffix] += val
    else:
        scal[name].append(val)


def pct(h, q):
    items = sorted(h.items()); tot = items[-1][1]; target = q * tot; lo, clo = 0.0, 0.0
    for le, c in items:
        if c >= target:
            return lo if le == float("inf") else (le if c == clo else lo + (le - lo) * (target - clo) / (c - clo))
        lo, clo = le, c
    return lo


def row(name, unit="s", mult=1.0, fmt="{:.2f}"):
    if name not in hist:
        return
    h = hist[name]; n = counts.get(name) or max(h.values())
    ps = [fmt.format(pct(h, q) * mult) for q in (0.5, 0.9, 0.99)]
    print(f"  {name:52s} mean {fmt.format(sums.get(name, 0) / n * mult)}{unit}  p50 {ps[0]}  p90 {ps[1]}  p99 {ps[2]}{unit}")


print("latency"); row("sglang:time_to_first_token_seconds"); row("sglang:inter_token_latency_seconds", "ms", 1000, "{:.1f}")
row("sglang:queue_time_seconds")
for k in sorted(hist):
    if k.startswith("sglang:per_stage_req_latency_seconds["):
        row(k)
print("sizes (tokens)")
for k in ("sglang:prompt_tokens_histogram", "sglang:uncached_prompt_tokens_histogram", "sglang:generation_tokens_histogram"):
    row(k, "", 1, "{:,.0f}")
tot = lambda k: sum(scal.get(k, []))
pt, ct, gt, nr = tot("sglang:prompt_tokens_total"), tot("sglang:cached_tokens_total"), tot("sglang:generation_tokens_total"), tot("sglang:num_requests_total")
if nr:
    print(f"requests {nr:,.0f} | prompt {pt / 1e9:.2f} B, {ct / pt:.1%} from cache | per request: prompt {pt / nr:,.0f}, "
          f"computed {(pt - ct) / nr:,.0f}, generated {gt / nr:,.0f} | aborted {tot('sglang:num_aborted_requests_total'):,.0f}, "
          f"retracted {tot('sglang:num_retracted_reqs'):,.0f}")
