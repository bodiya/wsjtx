"""Run analyze2 and add the frequency-offset results (results3.jsonl) to data.json."""
import json
import os

import analyze2  # noqa: F401  (writes data.json)

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data.json")))
import glob
recs = []
seen = set()
for path in [os.path.join(HERE, "results3.jsonl")] + sorted(glob.glob(os.path.join(HERE, "results3.*.jsonl"))):
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if r["key"] not in seen:
                seen.add(r["key"])
                recs.append(r)
CHS = ["awgn", "quiet", "moderate", "disturbed"]
TIMES = [30.0, 12.64, 4.48, 2.0]


def r1(x, k=2):
    return None if x is None else round(x, k)


def find(ch, T, group, drift=0.0):
    for r in recs:
        if r["ch"] == ch and abs(r["T"] - T) < 1e-6 and r["group"] == group and abs(r["drift"] - drift) < 1e-9:
            return r
    return None


def pt(r, lv):
    return None if r is None else dict(T=r["T"], R=r1(r["rate_bps"], 3), thr=r1(r[f"thr{lv}"]), baud=r1(r["baud"]), bw=r1(r["bandwidth"], 1))


freq = {"map": {}, "table": {}, "drift": {}, "n": len(recs)}
for ch in CHS:
    freq["map"][ch] = {}
    for lv in (50, 90):
        S = {}
        for g in ("nc_search", "coh_search", "coh_oracle", "bpsk_search", "bpsk_oracle"):
            S[g] = [p for p in (pt(find(ch, T, g), lv) for T in TIMES) if p]
            S[g].sort(key=lambda p: p["R"])
        freq["map"][ch][str(lv)] = S
    rows = []
    for name, g, src in (("8-FSK noncoherent, no offset", "fsk8", "map"), ("8-FSK noncoherent, offset ignored", "nc_none", "f"),
                         ("8-FSK noncoherent, offset searched", "nc_search", "f"), ("8-FSK coherent + DD, no offset", "fsk_coh_dd", "map"),
                         ("8-FSK coherent + DD, offset ignored", "coh_none", "f"), ("8-FSK coherent + DD, offset searched", "coh_search", "f"),
                         ("8-FSK coherent + DD, offset known", "coh_oracle", "f"),
                         ("BPSK MMSE + DD 18%, no offset", "bpsk_dd18", "map"), ("BPSK MMSE + DD 18%, offset searched", "bpsk_search", "f"),
                         ("BPSK MMSE + DD 18%, offset known", "bpsk_oracle", "f")):
        if src == "map":
            p = next((p for p in d["map"][ch]["50"][g] if abs(p["T"] - 12.64) < 1e-6), None)
            p9 = next((p for p in d["map"][ch]["90"][g] if abs(p["T"] - 12.64) < 1e-6), None)
            rows.append(dict(name=name, thr50=p["thr"] if p else None, thr90=p9["thr"] if p9 else None, key=g))
        else:
            r = find(ch, 12.64, g)
            rows.append(dict(name=name, thr50=r1(r["thr50"]) if r else None, thr90=r1(r["thr90"]) if r else None, key=g))
    freq["table"][ch] = rows
    freq["drift"][ch] = [dict(drift=dr, coh=(lambda r: r1(r["thr50"]) if r else None)(find(ch, 12.64, "coh_search", dr)),
                              coh90=(lambda r: r1(r["thr90"]) if r else None)(find(ch, 12.64, "coh_search", dr)),
                              nc=(lambda r: r1(r["thr50"]) if r else None)(find(ch, 12.64, "nc_search", dr)))
                         for dr in (0.0, 0.2, 0.5)]
d["freq"] = freq
d["meta"]["n_records"] += len(recs)
json.dump(d, open(os.path.join(HERE, "data.json"), "w"), separators=(",", ":"))
print("freq records", len(recs))
for ch in CHS:
    print(ch, [(r["key"], r["thr50"]) for r in freq["table"][ch]])
    print("   drift", freq["drift"][ch])
