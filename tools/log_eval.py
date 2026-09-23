#!/usr/bin/env python3
"""Production numbers from SGLang server logs (GLM-5.3-Flash with speculative decoding).

    python3 tools/log_eval.py --since "2026-09-19 01:00" --tz -3 /root/glm.log.1 /root/glm.log

Reads the TP0 "Decode batch" lines (one every decode_log_interval=40 passes) and "Prefill batch" lines (one per
prefill batch) and prints:
  1. decode per batch bucket: share of time, aggregate and per-request tok/s, acceptance, ms per verify step
  2. concurrency percentiles and a by-hour profile
  3. queueing: how often, and which resource was binding
  4. cold-prefill bursts (consecutive full chunks) and how long decode was stalled by them

Caveat for (4): with --prefill-decode-interval N > 0, decode passes run between prefill chunks but are only
logged every 40 passes, so bursts look capped at ~40/N chunks. Judge those servers by inter-token latency p99
(tools/metrics_summary.py) and by the count of long stalls, not by total stalled minutes.
"""
import argparse, collections, datetime as dt, re

RD = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0[^\]]*\] Decode batch, #running-req: (\d+), #full token: (\d+), '
                r'full token usage: ([\d.]+), mamba num: (\d+), mamba usage: ([\d.]+), accept len: ([\d.]+), accept rate: ([\d.]+), '
                r'cuda graph: (\w+), gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)')
RP = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0[^\]]*\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), '
                r'#cached-token: (\d+), full token usage: ([\d.]+), mamba usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)')
BUCKETS = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 48), (49, 64), (65, 96), (97, 512)]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--since", default="0000")
    ap.add_argument("--tz", type=int, default=0, help="hour offset for the by-hour table, e.g. -3")
    ap.add_argument("--chunk", type=int, default=4096, help="--chunked-prefill-size of the server")
    a = ap.parse_args()
    T = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    dec, ev = [], []
    for path in a.logs:
        with open(path, errors="replace") as f:
            for line in f:
                if "Decode batch" in line:
                    m = RD.match(line)
                    if m and m.group(1) >= a.since:
                        d = dict(t=T(m.group(1)), run=int(m.group(2)), kv=float(m.group(4)), mamba=float(m.group(6)),
                                 acc=float(m.group(7)), graph=m.group(9) == "True", tput=float(m.group(10)), q=int(m.group(11)))
                        dec.append(d); ev.append(("D", d))
                elif "Prefill batch" in line:
                    m = RP.match(line)
                    if m and m.group(1) >= a.since:
                        ev.append(("P", dict(t=T(m.group(1)), seqs=int(m.group(2)), new=int(m.group(3)),
                                             cached=int(m.group(4)), run=int(m.group(7)))))
    if not dec:
        raise SystemExit("no decode lines in range")
    dec.sort(key=lambda d: d["t"]); ev.sort(key=lambda e: e[1]["t"])
    hours = (dec[-1]["t"] - dec[0]["t"]).total_seconds() / 3600; days = hours / 24
    print(f"window {dec[0]['t']:%Y-%m-%d %H:%M} → {dec[-1]['t']:%Y-%m-%d %H:%M} ({hours:.1f} h) | {len(dec):,} decode samples")

    w = [min((dec[i + 1]["t"] - d["t"]).total_seconds(), 30.0) if i + 1 < len(dec) else 0 for i, d in enumerate(dec)]
    wt = sum(w)
    print("\n1) DECODE PER BATCH BUCKET")
    print("   bucket   samples  time    tok/s  p90 tok/s  per req  accept  ms/step  no graph")
    for lo, hi in BUCKETS:
        s = [(d, x) for d, x in zip(dec, w) if lo <= d["run"] <= hi]
        if not s:
            continue
        n = len(s); tp = [d["tput"] for d, _ in s]
        run = sum(d["run"] for d, _ in s) / n; acc = sum(d["acc"] for d, _ in s) / n; mean = sum(tp) / n
        steps = mean / (run * acc)
        print(f"   {lo:>3}-{hi:<4} {n:8,d} {sum(x for _, x in s) / wt:6.1%} {mean:8,.0f} {pct(tp, .9):9,.0f} "
              f"{sum(d['tput'] / d['run'] for d, _ in s) / n:8.0f} {acc:6.2f} {1000 / steps:8.1f} {sum(1 for d, _ in s if not d['graph']) / n:8.1%}")

    runs = [d["run"] for d in dec]
    print(f"\n2) CONCURRENCY p50 {pct(runs, .5)}  p90 {pct(runs, .9)}  p99 {pct(runs, .99)}  max {max(runs)}")
    by_h = collections.defaultdict(list)
    for d in dec:
        by_h[(d["t"].hour + a.tz) % 24].append(d)
    print(f"   hour(UTC{a.tz:+d})  mean batch  mean tok/s  queue>0")
    for h in sorted(by_h):
        s = by_h[h]
        print(f"   {h:02d}h        {sum(x['run'] for x in s) / len(s):6.1f}    {sum(x['tput'] for x in s) / len(s):8,.0f}   {sum(1 for x in s if x['q'] > 0) / len(s):6.1%}")

    q = [d for d in dec if d["q"] > 0]
    print(f"\n3) QUEUE: {len(q):,} of {len(dec):,} samples ({len(q) / len(dec):.2%}) had a queue")
    if q:
        cap = max(runs)
        print(f"   at the running cap: {sum(1 for d in q if d['run'] >= cap - 2) / len(q):.0%} | KDA pool ≥ 85%: "
              f"{sum(1 for d in q if d['mamba'] >= .85) / len(q):.0%} | KV ≥ 85%: {sum(1 for d in q if d['kv'] >= .85) / len(q):.0%}")
    print(f"   KDA pool use p50 {pct([d['mamba'] for d in dec], .5):.2f} p99 {pct([d['mamba'] for d in dec], .99):.2f} max {max(d['mamba'] for d in dec):.2f}"
          f" | KV in active use p50 {pct([d['kv'] for d in dec], .5):.2f} p99 {pct([d['kv'] for d in dec], .99):.2f}")

    bursts, cur = [], []
    for kind, e in ev:
        if kind == "P" and e["seqs"] == 1 and e["new"] == a.chunk:
            cur.append(e); continue
        if len(cur) >= 3:
            bursts.append((len(cur), (cur[-1]["t"] - cur[0]["t"]).total_seconds() + 0.2, max(x["run"] for x in cur), cur[0]["t"]))
        cur = []
    print(f"\n4) COLD-PREFILL BURSTS (≥ 3 consecutive full {a.chunk}-token chunks)")
    if bursts:
        dur = [b[1] for b in bursts]
        print(f"   {len(bursts):,} bursts ({len(bursts) / days:,.0f}/day) | stall p50 {pct(dur, .5):.1f}s p90 {pct(dur, .9):.1f}s "
              f"p99 {pct(dur, .99):.1f}s max {max(dur):.1f}s")
        print(f"   stalled time {sum(dur) / days / 60:.0f} min/day | user-seconds stalled {sum(b[1] * b[2] for b in bursts) / days / 3600:.1f} user-hours/day")
        print(f"   stalls ≥ 5 s with ≥ 5 users decoding: {sum(1 for b in bursts if b[1] >= 5 and b[2] >= 5) / days:,.0f}/day")
        per = sorted((b[1] - 0.2) / (b[0] - 1) for b in bursts if b[0] >= 10)
        if per:
            print(f"   seconds per chunk inside long bursts p50 {pct(per, .5):.3f} → ~{a.chunk / pct(per, .5):,.0f} tok/s cold prefill")


if __name__ == "__main__":
    main()
