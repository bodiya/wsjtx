"""Third sweep: frequency offset and drift. Every message gets a random offset uniform in
+-2 Hz (and optionally a linear drift); receivers either ignore it, search for it, or are
told the true value (oracle). Appends to results3.jsonl (resumable)."""
import json
import math
import os
import time
import warnings

import numpy as np

import wsim

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results3.jsonl")
ft8 = wsim.ft8_code(os.path.join(HERE, "ft8_ldpc.npz"))

PAYLOAD = 77
TIMES = [2.0, 4.48, 12.64, 30.0]
CHS = ["awgn", "quiet", "moderate", "disturbed"]
OFF = 2.0


def configs():
    for ch in CHS:
        for T in TIMES:
            yield dict(group="nc_search", T=T, ch=ch, off=OFF, drift=0.0)
            yield dict(group="coh_search", T=T, ch=ch, off=OFF, drift=0.0)
            yield dict(group="coh_oracle", T=T, ch=ch, off=OFF, drift=0.0)
            yield dict(group="bpsk_search", T=T, ch=ch, off=OFF, drift=0.0)
    for ch in CHS:
        yield dict(group="nc_none", T=12.64, ch=ch, off=OFF, drift=0.0)
        yield dict(group="coh_none", T=12.64, ch=ch, off=OFF, drift=0.0)
        for T in (12.64, 4.48):
            yield dict(group="bpsk_oracle", T=T, ch=ch, off=OFF, drift=0.0)
        for dr in (0.2, 0.5):
            yield dict(group="coh_search", T=12.64, ch=ch, off=OFF, drift=dr)
        yield dict(group="nc_search", T=12.64, ch=ch, off=OFF, drift=0.5)


def key(c):
    return json.dumps(c, sort_keys=True)


def build(c):
    g = c["group"]
    fading = c["ch"] != "awgn"
    if g == "nc_search":
        return wsim.FSKScheme(8, ft8, T=c["T"], freq="search"), (6.5 if not fading else 9.0)
    if g == "nc_none":
        return wsim.FSKScheme(8, ft8, T=c["T"]), (6.5 if not fading else 9.0)
    if g == "coh_search":
        return wsim.FSKScheme(8, ft8, T=c["T"], coherent=True, dd_iters=3, freq="search"), (4.5 if not fading else 7.5)
    if g == "coh_oracle":
        return wsim.FSKScheme(8, ft8, T=c["T"], coherent=True, dd_iters=3, freq="oracle"), (4.5 if not fading else 7.5)
    if g == "coh_none":
        return wsim.FSKScheme(8, ft8, T=c["T"], coherent=True, dd_iters=3), (6.5 if not fading else 9.0)
    if g == "bpsk_search":
        return wsim.BPSKScheme(ft8, T=c["T"], est="wiener", dd_iters=3, pilot_frac=0.18, freq="search"), (3.0 if not fading else 6.5)
    if g == "bpsk_oracle":
        return wsim.BPSKScheme(ft8, T=c["T"], est="wiener", dd_iters=3, pilot_frac=0.18, freq="oracle"), (3.0 if not fading else 6.5)
    raise ValueError(g)


def main():
    import glob
    shard, nshard = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARD", "1"))
    out = OUT if nshard == 1 else OUT.replace(".jsonl", f".{shard}.jsonl")
    done = set()
    for path in [OUT] + glob.glob(OUT.replace(".jsonl", ".*.jsonl")):
        if os.path.exists(path):
            for line in open(path):
                done.add(json.loads(line)["key"])
    todo = [c for c in configs() if key(c) not in done][shard::nshard]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    for i, c in enumerate(todo):
        t0 = time.time()
        sch, guess = build(c)
        ch = wsim.CHANNELS[c["ch"]]
        F = 500
        snrs, probs, thr = wsim.sweep(sch, ch, PAYLOAD, F=F, ebn0_guess=guess, pmax=0.97 if c["ch"] == "awgn" else 0.93,
                                      max_pts=26, seed=hash(key(c)) % (2**31), offset=c["off"], drift=c["drift"])
        R = PAYLOAD / c["T"]
        rec = dict(c, key=key(c), scheme=sch.name, baud=sch.baud, bandwidth=sch.bandwidth, rate_bps=R, frames=F,
                   snrs=snrs, probs=probs, thr50=thr[0.5], thr90=thr[0.9],
                   ebn0_50=None if thr[0.5] is None else thr[0.5] + 10 * math.log10(2500 / R),
                   ebn0_90=None if thr[0.9] is None else thr[0.9] + 10 * math.log10(2500 / R), secs=time.time() - t0)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i+1}/{len(todo)}] {c['group']:12s} T={c['T']:<6} {c['ch']:9s} drift={c['drift']} thr50={rec['thr50']} thr90={rec['thr90']} ({rec['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
