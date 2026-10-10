"""Second sweep: coherent receivers with MMSE + decision-directed channel estimation,
coherent detection of the FT8 waveform, pilot fraction, code rate, and the best combination.
Appends to results2.jsonl (resumable)."""
import json
import math
import os
import time
import warnings

import numpy as np

import wsim

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results2.jsonl")

ft8 = wsim.ft8_code(os.path.join(HERE, "ft8_ldpc.npz"))
pc = np.load(os.path.join(HERE, "peg_codes.npz"))
peg616 = wsim.LDPC(pc["H616"], "PEG LDPC(616,322)")
CODES = {"ft8": ft8, "peg616": peg616}
if "H273" in pc.files:
    CODES["r13"] = wsim.LDPC(pc["H273"], "PEG LDPC(273,91) rate 1/3")
    CODES["r14"] = wsim.LDPC(pc["H364"], "PEG LDPC(364,91) rate 1/4")
else:
    CODES["r13"] = wsim.peg_ldpc(273, 91, seed=5, name="PEG LDPC(273,91) rate 1/3")
    CODES["r14"] = wsim.peg_ldpc(364, 91, seed=5, name="PEG LDPC(364,91) rate 1/4")
    np.savez(os.path.join(HERE, "peg_codes.npz"), H174=pc["H174"], H616=pc["H616"], H273=CODES["r13"].H, H364=CODES["r14"].H)

PAYLOAD = 77
TIMES = [2.0, 4.48, 12.64, 30.0]
CHS = ["awgn", "quiet", "moderate", "disturbed"]


def configs():
    # A-D: the four main new receivers at every speed and channel
    for ch in CHS:
        for T in TIMES:
            yield dict(group="bpsk_wiener", code="ft8", T=T, ch=ch, payload=PAYLOAD, dd=0, pf=None)
            yield dict(group="bpsk_dd", code="ft8", T=T, ch=ch, payload=PAYLOAD, dd=3, pf=None)
            yield dict(group="fsk_coh", M=8, code="ft8", T=T, ch=ch, payload=PAYLOAD, dd=0)
            yield dict(group="fsk_coh_dd", M=8, code="ft8", T=T, ch=ch, payload=PAYLOAD, dd=3)
    # E: pilot fraction with DD
    for ch in CHS:
        for T in TIMES:
            for pf in (0.10, 0.18):
                yield dict(group="bpsk_dd", code="ft8", T=T, ch=ch, payload=PAYLOAD, dd=3, pf=pf)
    # F: lower code rates for noncoherent 8-FSK (bandwidth grows 1.5x / 2x)
    for ch in CHS:
        for T in (12.64, 4.48):
            for cd in ("r13", "r14"):
                yield dict(group="fsk_rate", M=8, code=cd, T=T, ch=ch, payload=PAYLOAD, dd=0)
    # G: best combination: BPSK + MMSE + DD, 18% pilots, 4-message block
    for ch in CHS:
        for T in TIMES:
            yield dict(group="best_long", code="peg616", T=4 * T, ch=ch, payload=4 * PAYLOAD, dd=3, pf=0.18)
    # H: DD iteration trace
    for ch in ("moderate", "disturbed"):
        for T in (4.48, 30.0):
            for dd in (1, 2):
                yield dict(group="bpsk_dd_trace", code="ft8", T=T, ch=ch, payload=PAYLOAD, dd=dd, pf=None)
                yield dict(group="fsk_coh_dd_trace", M=8, code="ft8", T=T, ch=ch, payload=PAYLOAD, dd=dd)
    # I: other tone counts with the coherent receiver
    for ch in CHS:
        for M in (4, 16):
            yield dict(group="fsk_coh_dd", M=M, code="ft8", T=12.64, ch=ch, payload=PAYLOAD, dd=3)
    # J: coherent 8-FSK + DD with the rate-1/3 code
    for ch in CHS:
        for T in (12.64, 4.48):
            yield dict(group="fsk_rate_coh", M=8, code="r13", T=T, ch=ch, payload=PAYLOAD, dd=3)


def key(c):
    return json.dumps(c, sort_keys=True)


def build(c):
    code = CODES[c["code"]]
    g = c["group"]
    fading = c["ch"] != "awgn"
    if g.startswith("bpsk") or g == "best_long":
        pf = c.get("pf") or wsim.SYNC_FRACTION
        return wsim.BPSKScheme(code, T=c["T"], est="wiener", dd_iters=c["dd"], pilot_frac=pf), (3.0 if not fading else 6.5)
    if g.startswith("fsk_coh") or g == "fsk_rate_coh":
        return wsim.FSKScheme(c["M"], code, T=c["T"], coherent=True, dd_iters=c["dd"]), (4.5 if not fading else 7.5)
    return wsim.FSKScheme(c["M"], code, T=c["T"]), (6.5 if not fading else 9.0)


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
        F = 250 if c["group"] == "best_long" else 500
        snrs, probs, thr = wsim.sweep(sch, ch, c["payload"], F=F, ebn0_guess=guess,
                                      pmax=0.97 if c["ch"] == "awgn" else 0.93, max_pts=26,
                                      seed=hash(key(c)) % (2**31))
        R = c["payload"] / c["T"]
        rec = dict(c, key=key(c), scheme=sch.name, code_name=sch.code.name, n=sch.code.n, k=sch.code.k,
                   baud=sch.baud, bandwidth=sch.bandwidth, nsym=sch.nsym, nsync=sch.nsync, rate_bps=R, frames=F,
                   snrs=snrs, probs=probs, thr50=thr[0.5], thr90=thr[0.9],
                   ebn0_50=None if thr[0.5] is None else thr[0.5] + 10 * math.log10(2500 / R),
                   ebn0_90=None if thr[0.9] is None else thr[0.9] + 10 * math.log10(2500 / R),
                   secs=time.time() - t0)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i+1}/{len(todo)}] {c['group']:16s} M={c.get('M','-')} code={c['code']:6s} T={c['T']:<6} {c['ch']:9s} "
              f"dd={c['dd']} pf={c.get('pf')} thr50={rec['thr50']} thr90={rec['thr90']} ({rec['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
