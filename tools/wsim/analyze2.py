"""Merge results.jsonl + results2.jsonl + theory.json into data.json for the report."""
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
recs1 = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]
recs2 = [json.loads(l) for l in open(os.path.join(HERE, "results2.jsonl"))] if os.path.exists(os.path.join(HERE, "results2.jsonl")) else []
theory = json.load(open(os.path.join(HERE, "theory.json")))

CHS = ["awgn", "quiet", "moderate", "disturbed"]
TIMES = [30.0, 12.64, 4.48, 2.0]


def r1(x, d=2):
    return None if x is None else round(x, d)


def point(r, level):
    thr = r[f"thr{level}"]
    return dict(T=r["T"], R=r1(r["rate_bps"], 3), thr=r1(thr), ebn0=r1(r[f"ebn0_{level}"]),
                baud=r1(r["baud"]), bw=r1(r["bandwidth"], 1), win=r.get("window", "-"),
                pf=r.get("pf"), dd=r.get("dd"), code=r.get("code_name"))


def find1(ch, T, group, **kw):
    for r in recs1:
        if r["ch"] == ch and abs(r["T"] - T) < 1e-6 and r["group"] == group and all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def find2(ch, T, group, **kw):
    for r in recs2:
        if r["ch"] == ch and abs(r["T"] - T) < 1e-6 and r["group"] == group and all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def best_pilot(ch, T, level):
    cand = [r for r in recs1 if r["ch"] == ch and abs(r["T"] - T) < 1e-6 and r["group"] == "pilot" and r[f"thr{level}"] is not None]
    return min(cand, key=lambda r: r[f"thr{level}"]) if cand else None


out = {"map": {}, "curves": {}, "theory": theory, "meta": {}, "stages": {}, "pilots": {}, "rate": {}, "tones_coh": {}}

# ---------------- map ----------------
for ch in CHS:
    out["map"][ch] = {}
    for level in (50, 90):
        S = {}
        rs = [r for r in recs1 if r["ch"] == ch]
        for M in (2, 4, 8, 16, 32):
            S[f"fsk{M}"] = sorted([point(r, level) for r in rs if r["group"] == "fsk" and r["M"] == M], key=lambda p: p["R"])
        for M in (4, 8, 16):
            S[f"long_fsk{M}"] = sorted([point(r, level) for r in rs if r["group"] == "long_fsk" and r["M"] == M], key=lambda p: p["R"])
        S["genie"] = sorted([point(r, level) for r in rs if r["group"] == "genie"], key=lambda p: p["R"])
        for g in ("pilot", "long_pilot"):
            byT = defaultdict(list)
            for r in rs:
                if r["group"] == g and r[f"thr{level}"] is not None:
                    byT[r["T"]].append(r)
            S[g] = sorted([point(min(c, key=lambda r: r[f"thr{level}"]), level) for c in byT.values()], key=lambda p: p["R"])
        rs2 = [r for r in recs2 if r["ch"] == ch]
        S["bpsk_wiener"] = sorted([point(r, level) for r in rs2 if r["group"] == "bpsk_wiener"], key=lambda p: p["R"])
        S["bpsk_dd"] = sorted([point(r, level) for r in rs2 if r["group"] == "bpsk_dd" and r.get("pf") is None], key=lambda p: p["R"])
        S["bpsk_dd18"] = sorted([point(r, level) for r in rs2 if r["group"] == "bpsk_dd" and r.get("pf") == 0.18], key=lambda p: p["R"])
        S["bpsk_dd10"] = sorted([point(r, level) for r in rs2 if r["group"] == "bpsk_dd" and r.get("pf") == 0.10], key=lambda p: p["R"])
        S["fsk_coh"] = sorted([point(r, level) for r in rs2 if r["group"] == "fsk_coh" and r["M"] == 8], key=lambda p: p["R"])
        S["fsk_coh_dd"] = sorted([point(r, level) for r in rs2 if r["group"] == "fsk_coh_dd" and r["M"] == 8], key=lambda p: p["R"])
        S["best_long"] = sorted([point(r, level) for r in rs2 if r["group"] == "best_long"], key=lambda p: p["R"])
        out["map"][ch][str(level)] = S

# ---------------- stages: closing the estimation gap ----------------
for ch in CHS:
    out["stages"][ch] = {}
    for T in TIMES:
        row = {}
        for level in (50, 90):
            bp = best_pilot(ch, T, level)
            g = find1(ch, T, "genie")
            w = find2(ch, T, "bpsk_wiener")
            d1 = find2(ch, T, "bpsk_dd_trace", dd=1)
            d2 = find2(ch, T, "bpsk_dd_trace", dd=2)
            d3 = find2(ch, T, "bpsk_dd", pf=None)
            f0 = find1(ch, T, "fsk", M=8)
            fc = find2(ch, T, "fsk_coh", M=8)
            fd1 = find2(ch, T, "fsk_coh_dd_trace", dd=1)
            fd2 = find2(ch, T, "fsk_coh_dd_trace", dd=2)
            fd3 = find2(ch, T, "fsk_coh_dd", M=8)
            k = f"thr{level}"
            row[str(level)] = dict(
                bpsk=[("Boxcar pilots", bp), ("MMSE", w), ("+DD 1", d1), ("+DD 2", d2), ("+DD 3", d3), ("Perfect channel", g)],
                fsk=[("Noncoherent", f0), ("Coherent, sync only", fc), ("+DD 1", fd1), ("+DD 2", fd2), ("+DD 3", fd3)],
            )
            for key in ("bpsk", "fsk"):
                row[str(level)][key] = [dict(name=n, thr=r1(r[k]) if r else None, win=(r.get("window") if r else None)) for n, r in row[str(level)][key]]
        out["stages"][ch][str(T)] = row

# ---------------- pilot fraction ----------------
for ch in CHS:
    out["pilots"][ch] = {}
    for T in TIMES:
        d = {}
        for level in (50, 90):
            pts = []
            for pf, lab in ((0.10, 0.10), (0.18, 0.18), (None, 21 / 79)):
                r = find2(ch, T, "bpsk_dd", pf=pf)
                if r:
                    pts.append(dict(pf=round(lab, 3), thr=r1(r[f"thr{level}"]), nsync=r["nsync"]))
            d[str(level)] = pts
        out["pilots"][ch][str(T)] = d

# ---------------- code rate ----------------
for ch in CHS:
    out["rate"][ch] = {}
    for T in (12.64, 4.48):
        rows = []
        for name, r in (("Noncoherent, rate 1/2 (FT8 code)", find1(ch, T, "fsk", M=8)),
                        ("Noncoherent, rate 1/3", find2(ch, T, "fsk_rate", code="r13")),
                        ("Noncoherent, rate 1/4", find2(ch, T, "fsk_rate", code="r14")),
                        ("Coherent + DD, rate 1/2 (FT8 code)", find2(ch, T, "fsk_coh_dd", M=8)),
                        ("Coherent + DD, rate 1/3", find2(ch, T, "fsk_rate_coh", code="r13"))):
            if r:
                rows.append(dict(name=name, bw=r1(r["bandwidth"], 1), baud=r1(r["baud"]), thr50=r1(r["thr50"]), thr90=r1(r["thr90"]), coh="Coherent" in name))
        out["rate"][ch][str(T)] = rows

# ---------------- tone count with the coherent receiver ----------------
for ch in CHS:
    rows = []
    for M in (4, 8, 16):
        nc = find1(ch, 12.64, "fsk", M=M)
        co = find2(ch, 12.64, "fsk_coh_dd", M=M)
        rows.append(dict(M=M, bw=r1(nc["bandwidth"], 1) if nc else None, noncoh50=r1(nc["thr50"]) if nc else None, coh50=r1(co["thr50"]) if co else None,
                         noncoh90=r1(nc["thr90"]) if nc else None, coh90=r1(co["thr90"]) if co else None))
    out["tones_coh"][ch] = rows

# ---------------- decode-probability curves ----------------
for ch in CHS:
    out["curves"][ch] = {}
    for T in (2.0, 4.48, 12.64, 30.0):
        cur = {}
        for r in recs1:
            if r["ch"] != ch:
                continue
            if r["group"] == "fsk" and r["T"] == T:
                cur[f"fsk{r['M']}"] = dict(s=r["snrs"], p=r["probs"])
            elif r["group"] == "genie" and r["T"] == T:
                cur["genie"] = dict(s=r["snrs"], p=r["probs"])
            elif r["group"] == "long_fsk" and r["M"] == 8 and abs(r["T"] - 4 * T) < 1e-6:
                cur["long_fsk8"] = dict(s=r["snrs"], p=r["probs"])
        bp = best_pilot(ch, T, 50)
        if bp:
            cur["pilot"] = dict(s=bp["snrs"], p=bp["probs"], win=bp.get("window"))
        for g, kk, kw in (("bpsk_dd", "bpsk_dd18", dict(pf=0.18)), ("fsk_coh_dd", "fsk_coh_dd", dict(M=8)), ("best_long", "best_long", {})):
            r = find2(ch, 4 * T if g == "best_long" else T, g, **kw)
            if r:
                cur[kk] = dict(s=r["snrs"], p=r["probs"])
        out["curves"][ch][str(T)] = cur

# ---------------- FT8 budget + published ----------------
f8 = find1("awgn", 12.64, "fsk", M=8)
sync_db = 10 * math.log10(79 / 58)
crc_db = 10 * math.log10(91 / 77)
code_rate = 91 / 174
rates = theory["rates"]
i = max(j for j, r in enumerate(rates) if r <= code_rate)


def at_rate(curve):
    return curve[i] + (curve[i + 1] - curve[i]) * (code_rate - rates[i]) / (rates[i + 1] - rates[i])


lim8 = at_rate(theory["curves"]["awgn_fsk8"])
lim8c = at_rate(theory["curves"]["awgn_cfsk8"])
shannon = theory["shannon_ebn0"]
sim_eb = f8["ebn0_50"]
out["meta"]["ft8_budget"] = dict(shannon=r1(shannon), noncoherent_8fsk=r1(lim8 - shannon), finite_len=r1(sim_eb - sync_db - crc_db - lim8),
                                 crc=r1(crc_db), sync=r1(sync_db), total=r1(sim_eb), sim_thr=r1(f8["thr50"]), lim8=r1(lim8),
                                 lim8_coherent=r1(lim8c), coherent_part=r1(lim8 - lim8c))
out["meta"]["published"] = [dict(name="FT8", T=12.64, R=r1(77 / 12.64, 3), thr=-21.0), dict(name="FT4", T=4.48, R=r1(77 / 4.48, 3), thr=-17.5)]
out["meta"]["n_records"] = len(recs1) + len(recs2)
json.dump(out, open(os.path.join(HERE, "data.json"), "w"), separators=(",", ":"))
print("records", len(recs1), "+", len(recs2))
for ch in CHS:
    S = out["map"][ch]["50"]
    for k in ("fsk8", "fsk_coh", "fsk_coh_dd", "pilot", "bpsk_wiener", "bpsk_dd", "bpsk_dd18", "bpsk_dd10", "best_long", "genie"):
        if S.get(k):
            print(f"  {ch:9s} {k:12s}", [p["thr"] for p in S[k]])
