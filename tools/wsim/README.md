# Weak-signal HF mode simulator

Monte Carlo comparison of noncoherent M-FSK, coherent M-FSK (same waveform,
phase-tracking receiver) and pilot-aided coherent BPSK on AWGN and
Watterson-model HF channels (ITU-R F.1487 mid-latitude quiet, moderate,
disturbed), all carrying a 77-bit message + 14-bit CRC with FT8's (174,91)
LDPC code. SNR is referenced to 2500 Hz, like WSJT-X.

Second pass adds MMSE (Wiener) channel estimation with per-message Doppler
selection, decision-directed re-estimation from the decoder's soft output,
a coherent receiver for the FT8 waveform, pilot-fraction and code-rate sweeps.

Third pass adds a random frequency offset (and optional drift) to every message
and a joint offset/Doppler likelihood search on the sync tones, with candidate
retries and decision-directed refinement, so the coherent receivers find the
frequency themselves.

## Files

| File | What it does |
|---|---|
| `wsim.py` | The simulator: LDPC encode + belief-propagation decode (Numba), PEG code construction, Watterson channel, M-FSK and BPSK modems, SNR sweeps |
| `parse_ft8.py` | Extracts FT8's LDPC matrices from ft8_lib's `ft8/constants.c` into `ft8_ldpc.npz` (already included) |
| `sweep.py` | First pass: FSK tone counts, boxcar-pilot BPSK, 4-message blocks → `results.jsonl` |
| `sweep2.py` | Second pass: MMSE + decision-directed receivers, coherent FSK, pilot fraction, code rate → `results2.jsonl` |
| `sweep3.py` | Third pass: frequency offset ±2 Hz, searched / ignored / known, drift → `results3.jsonl` |
| `theory.py` | Mutual-information limits for noncoherent M-FSK and coherent BPSK → `theory.json` |
| `analyze2.py`, `analyze3.py` | Condense the result files into `data.json` for the report |
| `results.jsonl`, `results2.jsonl`, `results3.jsonl` | Raw results: every SNR point and decode probability for 500 configurations |

## Running

```
pip install numpy scipy numba
python3 sweep.py      # ~1.5 hours on 2 cores
python3 sweep2.py     # ~2.5 hours
python3 sweep3.py     # ~2 hours (SHARD=0 NSHARD=2 / SHARD=1 NSHARD=2 to run two at once)
python3 theory.py     # ~10 minutes
python3 analyze3.py
```

## Trying your own mode

```python
import numpy as np, wsim
code = wsim.ft8_code("ft8_ldpc.npz")                 # or wsim.peg_ldpc(n, k)
mode = wsim.FSKScheme(M=8, code=code, T=12.64)       # tones, code, seconds per message
# mode = wsim.FSKScheme(8, code, T=12.64, coherent=True, dd_iters=3, freq="search")   # coherent FT8-waveform receiver that finds the frequency
# snrs, probs, thr = wsim.sweep(mode, ch, payload_bits=77, offset=2.0, drift=0.2)       # random offset +-2 Hz, drift +-0.2 Hz per message
# mode = wsim.BPSKScheme(code, T=4.48, est="wiener", dd_iters=3, pilot_frac=0.18)
ch = wsim.CHANNELS["moderate"]                       # awgn | quiet | moderate | disturbed
snrs, probs, thr = wsim.sweep(mode, ch, payload_bits=77)
print(thr)   # SNR (dB, 2500 Hz) for 50% and 90% decodes
```

Custom channels: `wsim.Channel("name", delay_s=0.002, doppler_hz=2.0)`.

## Assumptions

- Perfect time sync. Frequency is known unless `freq="search"` (then found from the sync tones) or `offset` is given. Sync or pilot symbols still cost FT8's share of air time (21 of 79 symbols).
- Receiver knows noise power and average signal level.
- BP decoding only (no OSD, no a-priori hypotheses). The FT8-geometry case lands at −19.7 dB vs WSJT-X's published −21 dB.
- Coherent receivers assume one complex gain across the signal bandwidth (no per-tone estimation).

FT8 LDPC matrices are from ft8_lib by Kārlis Goba (MIT licence), which follows the WSJT-X definition.
