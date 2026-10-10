"""Theoretical limits: BICM capacity of noncoherent M-FSK vs coherent BPSK.

For each modulation and channel, find the minimum Eb/N0 at which a code of
rate r (info bits per coded bit) can work, by Monte-Carlo estimating the bit-
interleaved generalised mutual information (what an LDPC code fed with bit
LLRs can achieve). Channels: AWGN and ideally-interleaved Rayleigh fading
(fast fading = the most time diversity a long message could ever get).
Writes theory.json.
"""
import json
import math
import os

import numpy as np

rng = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))


def lse(a, axis=-1):
    mx = a.max(axis=axis, keepdims=True)
    return (mx + np.log(np.exp(a - mx).sum(axis=axis, keepdims=True))).squeeze(axis)


def gmi_fsk(M, esn0, fading, N):
    bps = int(math.log2(M))
    es = 10 ** (esn0 / 10)
    s = rng.integers(0, M, N)
    noise = (rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))) / math.sqrt(2)
    h = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / math.sqrt(2) if fading else np.ones(N)
    y = noise
    y[np.arange(N), s] += math.sqrt(es) * h
    E = np.abs(y) ** 2
    if fading:
        m = E * es / (1 + es)
    else:
        x = 2 * math.sqrt(es) * np.sqrt(E)
        m = np.where(x < 50, np.log(np.i0(np.minimum(x, 50))), x - 0.5 * np.log(2 * np.pi * np.maximum(x, 1e-9)))
    labels = (np.arange(M)[:, None] >> np.arange(bps)[::-1]) & 1
    tot = 0.0
    for b in range(bps):
        z = labels[:, b] == 0
        L = lse(m[:, z]) - lse(m[:, ~z])
        sgn = 1 - 2 * labels[s, b]
        tot += 1 - np.mean(np.logaddexp(0, -sgn * L)) / math.log(2)
    return tot  # bits per symbol


def gmi_bpsk(esn0, fading, N):
    es = 10 ** (esn0 / 10)
    x = 1 - 2 * rng.integers(0, 2, N)
    h = np.abs((rng.standard_normal(N) + 1j * rng.standard_normal(N)) / math.sqrt(2)) if fading else np.ones(N)
    y = math.sqrt(es) * h * x + rng.standard_normal(N) / math.sqrt(2)
    L = 4 * math.sqrt(es) * h * y  # perfect channel knowledge
    return 1 - np.mean(np.logaddexp(0, -x * L)) / math.log(2)


def min_ebn0_curve(fn, bps, rates):
    grid = np.arange(-22, 24.01, 0.5)
    caps = np.array([fn(g) for g in grid])
    caps = np.maximum.accumulate(caps)
    out = []
    for r in rates:
        target = r * bps
        if target >= caps[-1]:
            out.append(None)
            continue
        i = np.searchsorted(caps, target)
        if i == 0:
            out.append(None)
            continue
        e = grid[i - 1] + (target - caps[i - 1]) / (caps[i] - caps[i - 1]) * 0.5
        out.append(float(e - 10 * math.log10(target)))  # Eb/N0 = Es/N0 / (r*bps)
    return out


def main():
    rates = [round(x, 3) for x in np.arange(0.05, 0.951, 0.05)]
    res = {"rates": rates, "curves": {}}
    for fading in (False, True):
        chn = "rayleigh" if fading else "awgn"
        for M in (2, 4, 8, 16, 32, 64):
            N = 40000 if M <= 16 else 16000
            curve = min_ebn0_curve(lambda g: gmi_fsk(M, g, fading, N), int(math.log2(M)), rates)
            res["curves"][f"{chn}_fsk{M}"] = curve
            best = min((v, r) for v, r in zip(curve, rates) if v is not None)
            print(f"{chn:8s} {M:2d}-FSK noncoherent: min Eb/N0 {best[0]:.2f} dB at rate {best[1]}", flush=True)
        curve = min_ebn0_curve(lambda g: gmi_bpsk(g, fading, 200000), 1, rates)
        res["curves"][f"{chn}_bpsk"] = curve
        best = min((v, r) for v, r in zip(curve, rates) if v is not None)
        print(f"{chn:8s} coherent BPSK (perfect CSI): min Eb/N0 {best[0]:.2f} dB at rate {best[1]}", flush=True)
    res["shannon_ebn0"] = 10 * math.log10(math.log(2))
    json.dump(res, open(os.path.join(HERE, "theory.json"), "w"))


if __name__ == "__main__":
    main()
