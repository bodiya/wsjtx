"""Run the full comparison sweep; results appended to results.jsonl (resumable)."""
import json
import math
import os
import sys
import time
import warnings

import numpy as np

import wsim

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.jsonl")

ft8 = wsim.ft8_code(os.path.join(HERE, "ft8_ldpc.npz"))
pc = np.load(os.path.join(HERE, "peg_codes.npz"))
peg616 = wsim.LDPC(pc["H616"], "PEG LDPC(616,322)")

PAYLOAD = 77  # FT8-style message; +14 CRC bits = 91 code info bits
TIMES = [2.0, 4.48, 12.64, 30.0]
CHS = ["awgn", "quiet", "moderate", "disturbed"]
WINDOWS = [0.3, 1.0, 3.0, None]


def configs():
    for ch in CHS:  # main family first so partial results are useful
        for T in TIMES:
            for M in [2, 4, 8, 16, 32]:
                yield dict(group="fsk", M=M, code="ft8", T=T, ch=ch, payload=PAYLOAD)
            yield dict(group="genie", code="ft8", T=T, ch=ch, payload=PAYLOAD)
            for w in ([None] if ch == "awgn" else WINDOWS):
                yield dict(group="pilot", code="ft8", T=T, ch=ch, window=w, payload=PAYLOAD)
    for ch in CHS:
        for T in TIMES:
            for M in [4, 8, 16]:
                yield dict(group="long_fsk", M=M, code="peg616", T=4 * T, ch=ch, payload=4 * PAYLOAD)
            for w in ([None] if ch == "awgn" else WINDOWS):
                yield dict(group="long_pilot", code="peg616", T=4 * T, ch=ch, window=w, payload=4 * PAYLOAD)


def key(c):
    return json.dumps(c, sort_keys=True)


def build(c):
    code = ft8 if c["code"] == "ft8" else peg616
    g = c["group"]
    if g in ("fsk", "long_fsk"):
        return wsim.FSKScheme(c["M"], code, T=c["T"]), (6.5 if c["ch"] == "awgn" else 9.0)
    if g == "genie":
        return wsim.BPSKScheme(code, T=c["T"], est="genie"), (2.5 if c["ch"] == "awgn" else 6.0)
    return wsim.BPSKScheme(code, T=c["T"], est="pilot", window=c.get("window")), (3.0 if c["ch"] == "awgn" else 8.0)


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            done.add(json.loads(line)["key"])
    todo = [c for c in configs() if key(c) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    for i, c in enumerate(todo):
        t0 = time.time()
        sch, guess = build(c)
        ch = wsim.CHANNELS[c["ch"]]
        F = 300 if c["group"] in ("pilot", "long_pilot") and c.get("window") is not None else 500
        snrs, probs, thr = wsim.sweep(sch, ch, c["payload"], F=F, ebn0_guess=guess,
                                      pmax=0.97 if c["ch"] == "awgn" else 0.93, max_pts=26,
                                      seed=hash(key(c)) % (2**31))
        R = c["payload"] / c["T"]
        rec = dict(c, key=key(c), scheme=sch.name, n=sch.code.n, k=sch.code.k, baud=sch.baud,
                   bandwidth=sch.bandwidth, nsym=sch.nsym, rate_bps=R, frames=F,
                   snrs=snrs, probs=probs, thr50=thr[0.5], thr90=thr[0.9],
                   ebn0_50=None if thr[0.5] is None else thr[0.5] + 10 * math.log10(2500 / R),
                   ebn0_90=None if thr[0.9] is None else thr[0.9] + 10 * math.log10(2500 / R),
                   secs=time.time() - t0)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i+1}/{len(todo)}] {c['group']:10s} M={c.get('M','-')} T={c['T']:<6} {c['ch']:9s} "
              f"w={c.get('window','-')} thr50={rec['thr50']} thr90={rec['thr90']} ({rec['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
