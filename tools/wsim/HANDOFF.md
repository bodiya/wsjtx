# Coherent FT8 Handoff

As of 2026-10-10. Brief for prototyping a coherent FT8 decoder in WSJT-X (or another FT8 decoder), from a simulation study done in a Claude chat on 9–10 October 2026. The change is receive-side only: the signal on the air stays standard FT8, so it is fully backward compatible.

Companion material:

| Item | Where | What it holds |
| --- | --- | --- |
| Interactive report | https://claude.ai/artifact/MVzoWbo79iRMHCV4sJUDPH | All results, charts and method, 500 simulated configurations; section 2 is coherent FT8, section 3 the frequency search, section 13 the method |
| Simulator package | this directory (`weak-signal-sim`) | `wsim.py` (the reference implementation), the three sweep scripts, raw results, `README.md` |
| This brief as a shared doc | https://claude.ai/artifact/CXoUFfezXXxofao5fwbUM6 | Same text, editable and commentable |

Claude Code on the web and in the desktop app reads claude.ai artifact links; the terminal CLI may not, which is why this file and `wsim.py` live in the repo.

## 1. Results to carry over

Coherent detection of the unchanged FT8 waveform decodes 1.2–2.8 dB deeper than noncoherent detection at FT8's 12.6 s per message, with the receiver finding the signal frequency on its own. SNR is in a 2500 Hz noise bandwidth (the WSJT-X convention); thresholds are where 50% of messages decode.

| Channel (ITU-R F.1487) | Noncoherent, own fine sync (dB) | Coherent + DD, own frequency search (dB) | Gain (dB) | Coherent, frequency known (dB) |
| --- | --- | --- | --- | --- |
| AWGN | −19.7 | −21.9 | 2.2 | −21.8 |
| Quiet (0.5 ms, 0.1 Hz) | −17.0 | −19.8 | 2.8 | −19.7 |
| Moderate (1 ms, 0.5 Hz) | −16.9 | −18.6 | 1.7 | −18.8 |
| Disturbed (2 ms, 1 Hz) | −16.8 | −18.1 | 1.2 | −18.3 |

What else the prototype should expect:

- The simulated noncoherent decoder reaches −19.7 dB on AWGN against WSJT-X's published −21 dB. The 1.3 dB difference is WSJT-X's ordered-statistics and a-priori decoding, which the simulation omits and which apply to either receiver. A coherent WSJT-X should land near −23 dB on AWGN if the gains stack.
- Mutual-information limits put the coherent-versus-noncoherent difference for 8-FSK at FT8's code rate at 2.2 dB on AWGN and 2.7 dB on Rayleigh fading: the simulation is at the limit on steady paths and estimation-limited on fading ones.
- Ignoring a frequency offset of up to ±2 Hz costs 1.6–2.1 dB and lands the coherent receiver exactly at the noncoherent level, not below it: the metric degrades gracefully.
- Linear drift of ±0.2 Hz over a message (about 1 Hz per minute) costs nothing; ±0.5 Hz costs 0.5 dB at 50% decodes and 1.7 dB at 90% on AWGN.
- Decision-directed passes matter on fading paths: at 12.6 s on the moderate path the sync tones alone gain 0.2 dB and the passes add 1.7 dB. Three passes capture the gain; the third is within noise of the second.
- At 2 s per message on the disturbed path the coherent receiver is 0.3 dB worse than noncoherent, because it assumes one channel gain across the 316 Hz signal. Not a concern for FT8's 50 Hz signal.
- Sampling noise in the thresholds is about ±0.2 dB on AWGN and ±0.4 dB on the quiet channel, whose decode curves are flat.

## 2. The coherent receiver, step by step

The receiver keeps the complex tone correlations that the noncoherent decoder reduces to magnitudes, estimates the channel's complex gain over time from the Costas sync tones, scores tones with a metric that trusts that estimate only as far as its error allows, and after a first decode uses the decoder's soft output as extra pilots. Everything below is what `wsim.py` does. Symbol period 0.16 s, 79 symbols, sync at symbols 0–6, 36–42 and 72–78 (0-based).

1. **Inputs.** Complex baseband samples at the decoder's downsampled rate (32 per symbol at 200 Hz in WSJT-X), already time-synced, with a coarse frequency within about ±2 Hz from the existing sync.
2. **Tone correlations at an offset δ.** Derotate the samples by exp(−j2πδt), then per symbol take the 32-point FFT and keep bins 0–7: Y_k(n). This is what the noncoherent decoder already computes before taking magnitudes. Noise variance per bin σ²: 7 of every 8 bins are noise, so σ² ≈ median(|Y|² over all bins and symbols) / ln 2.
3. **Sync observations.** z_i = Y at the known Costas tone of sync symbol i, for the 21 sync symbols. Each is h(t_i)·exp(j2πδ_true t_i) plus noise.
4. **Channel model.** h(t) is a unit-variance complex Gaussian process with autocorrelation R(τ) = exp(−2π²σ_d²τ²), σ_d = Doppler spread / 2 (Gaussian Doppler spectrum, the Watterson model). Hypothesis bank for the spread: 0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1, 1.5, 2 Hz. The static hypothesis makes R all ones, which turns the smoother into a coherent average over the whole message.
5. **Joint offset and Doppler search.** For each hypothesis d, A_d = R_d + σ²I over the 21 sync times and M_d = A_d⁻¹. With W_ik = conj(z_i)·z_k, the log-likelihood of offset Δ under d is

       ll(Δ, d) = −Re Σ_ik W_ik · M_d,ik · exp(j2πΔ(t_i − t_k)) − log det A_d

   which is one matrix product per hypothesis over a grid of Δ. Grid: ±3.1 Hz in steps of min(0.025, 0.2/T) Hz (16 mHz at T = 12.64 s), then a fine pass at step/12 within ±1.5 steps of each candidate. The three equally spaced sync blocks give near-equal grating lobes at ±2/T, so keep the five strongest local maxima separated by at least 1/T as candidates, best first. Pick the Doppler hypothesis over the full bank at the best Δ by marginal likelihood, then recompute Y at that Δ (step 2).
6. **MMSE channel smoothing.** Stack one observation per symbol: the sync tones with variance σ², data symbols with variance 10⁶ (unobserved). With V = diag of those variances,

       ĥ = R (R + V)⁻¹ z,     σ_e,n² = 1 − [R (R + V)⁻¹ R]_nn

   Blocks of 160 symbols with 32 overlap keep this linear in message length; FT8 fits in one block (79 × 79).
7. **Partially coherent tone metric.** Given the estimate and its error variance at symbol n, tone k on means Y_k ~ CN(ĥ, σ² + σ_e²) and off means CN(0, σ²), so

       m_k(n) = 2·Re(conj(ĥ_n)·Y_k(n)) / (σ² + σ_e,n²) + |Y_k(n)|² · (1/σ² − 1/(σ² + σ_e,n²))

   This is pure coherent detection when σ_e² → 0 and pure noncoherent when σ_e² → ∞, which is why a bad estimate costs the gain but never more.
8. **Bit LLRs.** For each of the 3 bits per symbol, L_b = logsumexp of m_k over tones whose Gray label has bit b = 0, minus the same over tones with bit b = 1. Use FT8's Gray map (tone index → bits: 0,1,3,2,5,6,4,7). These are true log-likelihood ratios, so they go to the LDPC decoder without the scaling the magnitude metrics need.
9. **Decode.** Belief propagation as today, then OSD and a-priori passes as today. Keep the posterior LLRs (channel LLR plus all incoming check messages per bit); the decision-directed pass needs them.
10. **Decision-directed pass, up to three, only for messages that fail the syndrome or CRC.** From the posterior bit LLRs form tone probabilities p_k(n) proportional to the product over bits of P(bit = label bit), then q = Σp² and q₃ = Σp³. The soft observation for symbol n is z_n = Σ_k p_k Y_k / q with noise variance v_n = (σ²q + max(q₃ − q², 0)) / q²; skip symbols with q < 10⁻³. Refine the offset with the same likelihood over all symbols (per-message M from these variances), ±0.3 Hz at 5 mHz then ±10 mHz at 1 mHz, recompute Y, re-smooth with sync plus soft observations, re-score, re-decode.
11. **Candidate retry.** A message still failing after the passes restarts from step 6 at the next frequency candidate from step 5.

Two points that are easy to get wrong: the derotation in step 10 is relative to the current estimate, so the refinement models only the residual (an early version applied it twice and drifted); and the Doppler choice must be re-done at the chosen offset, since a wrong offset looks like fast fading.

## 3. Reference implementation map

`wsim.py` is the executable specification; every step above maps to one function. About 900 lines of NumPy with one Numba kernel.

| Step | Function in `wsim.py` | Notes |
| --- | --- | --- |
| Waveform | `FSKScheme.modulate` | Tones switch at symbol boundaries with integer cycles per symbol; no GFSK smoothing (see pitfalls) |
| Channel | `apply_channel`, `fading_taps`, `CHANNELS` | Two-path Watterson model with true fractional delay; `offset` and `drift` arguments |
| Tone correlations at an offset | `FSKScheme.bins` | Derotate, per-symbol FFT, keep M bins |
| Sync observations | `FSKScheme._sync_obs` | Costas tone per sync symbol |
| Channel autocorrelation | `gauss_corr` | Gaussian Doppler spectrum; `DOPPLER_HYPS` is the bank |
| Joint offset/Doppler search | `search_offset` | Returns candidate list and Doppler per message |
| Doppler choice at a fixed offset | `select_doppler` | Marginal likelihood over the bank |
| MMSE smoothing | `wiener_smooth` | Blocked; a shared-weights fast path for the pilot-only pass |
| Partially coherent metric and bit LLRs | `FSKScheme._coherent_llr`, `FSKScheme._bit_llrs` | The metric formula is the one line in `_coherent_llr` |
| LDPC decode with posteriors | `LDPC.decode`, `_bp_decode` | Returns hard bits and posterior LLRs; `LDPC.syndrome_ok` |
| Decision-directed pass | `FSKScheme.refine`, `refine_offset` | Soft tone probabilities, soft observations, offset refinement |
| Candidate retry and the overall loop | `run_point` | Pass 0, DD passes, retries on the next candidate |
| FT8's code | `ft8_code`, `parse_ft8.py` | Parity-check and generator matrices parsed from ft8_lib's `constants.c` |

The same machinery serves `BPSKScheme`, the purpose-built mode in the report, not needed for the WSJT-X prototype.

A ten-second check that the reference behaves, from this directory (needs `numpy`, `scipy`, `numba`):

```python
import numpy as np, wsim
code = wsim.ft8_code("ft8_ldpc.npz")
ch = wsim.CHANNELS["awgn"]
nc = wsim.FSKScheme(8, code, T=12.64, freq="search")
co = wsim.FSKScheme(8, code, T=12.64, coherent=True, dd_iters=3, freq="search")
rng = lambda: np.random.default_rng(1)
for snr in (-22, -21, -20):
    a = wsim.run_point(nc, ch, snr, 300, rng(), offset=2.0) / 300
    b = wsim.run_point(co, ch, snr, 300, rng(), offset=2.0) / 300
    print(snr, "noncoherent", round(a, 2), "coherent", round(b, 2))
```

Expected: noncoherent near 0 at −22 and −21 dB and about 0.3 at −20; coherent about 0.4, 0.9 and 0.97.

To inspect what the search estimated, call `run_point(..., return_state=True)`; the state carries `df` (estimated offsets), `true_offset`, `dop` (chosen Doppler) and `cands` per message.

## 4. Where it plugs into WSJT-X

The change lives entirely inside the per-candidate FT8 decode: keep the complex per-symbol spectra the decoder already computes, and add a coherent scoring path that runs when the standard path fails. File and routine names below are from memory of the WSJT-X 2.x source tree and must be confirmed with a search of the checkout before editing (`rg -n 'four2a|bmeta|bpdecode174_91' lib/ft8`).

| File (under `lib/ft8/`) | Role today | Change |
| --- | --- | --- |
| `ft8_decode.f90` | Candidate loop: `sync8` for candidates, `ft8b` per candidate, `subtractft8` after each decode, several passes | None at first; later, feed the coherent channel estimate to subtraction |
| `sync8.f90`, `sync8d.f90` | Coarse sync (frequency, DT) and fine sync correlation on the Costas arrays | None; its frequency estimate is the coarse input to the search |
| `ft8_downsample.f90` | Mixes the candidate to baseband and downsamples to 200 Hz complex (`cd0`, 32 samples per symbol) | None; `cd0` is the input to the coherent path |
| `ft8b.f90` | Per-symbol 32-point FFT (`four2a`) into `csymb`, magnitudes `s8`, bit metrics `bmeta`/`bmetb`/..., normalisation, AP masks, `bpdecode174_91`, `osd174_91`, CRC, unpack | Keep `csymb` complex; after the existing decode attempts fail, call the new coherent routine with `cd0`, the sync index and the frequency, and run the same decode/CRC/unpack on its LLRs |
| `bpdecode174_91.f90` | BP decoder; computes posterior LLRs internally and returns hard bits | Expose the posterior LLRs as an optional output for the decision-directed pass |
| new `ft8_coherent.f90` | — | Steps 2–11 of the receiver |
| `gen_ft8wave.f90`, `gfsk_pulse.f90` | Transmit waveform, GFSK with BT = 2.0 | None for the first prototype; later, correlate against the true pulse shape |
| `subtractft8.f90` | Regenerates a decoded signal and subtracts it | Later: use the per-symbol channel estimate instead of a single complex amplitude |
| `ft8sim.f90` | Test-signal generator with SNR, Doppler spread and delay spread | The validation harness, unchanged |

Design choices for the prototype:

- Run the standard noncoherent path first and the coherent path only on failure. Every message that decodes today decodes identically, so the gain is purely additive and regression testing is simple.
- The noise variance per bin can come from the median of |Y|² over all bins (step 2) or from the decoder's existing baseline estimate; check that the two agree within a dB before trusting the metric's absolute LLR scale.
- The coherent LLRs are proper log-likelihood ratios. Do not apply the normalisation the magnitude metrics get; if the decoder has a fixed LLR scale factor, test with and without it.
- Linear algebra: the search needs a 21 × 21 real symmetric inverse per Doppler hypothesis (five in the coarse pass, ten for the final choice) and the smoother a 79 × 79 real symmetric solve per pass. A hand-written Cholesky is enough. The offset grid is about 400 points times 441 pair terms per hypothesis.
- Expect roughly 2–3× the decode time of the noncoherent path per candidate that reaches it, on the candidates that fail today. In the simulation the candidate retries roughly doubled time at threshold.
- Keep the whole thing behind a switch (a new depth level or a config flag) so A/B runs on the same recordings are one command apart.

## 5. Validation plan

The acceptance test is a decode-rate curve from `ft8sim` recordings, before and after, with the noncoherent path bit-exact.

1. **Regression first.** With the coherent path switched off, the patched `jt9` must produce byte-identical decode output on the WSJT-X sample files and on a set of `ft8sim` files. With it on, every message the old decoder found must still be found with the same text.
2. **AWGN curve.** `ft8sim` generates batches at a chosen SNR with no fading (`ft8sim "<message>" <f0> <DT> <fdop> <del> <width> <nfiles> <snr>`; confirm the argument order in `ft8sim.f90`). Generate 200 files at each of −19, −20, −21, −22, −23, −24 dB with `fdop` = 0 and `del` = 0, run `jt9 -8 -d 3` on each batch with the coherent path off and on, and count decodes. Target: the 50% point moves down by about 2 dB.
3. **Fading curves.** Repeat with the ITU profiles through `fdop` and `del`: 0.1 Hz / 0.5 ms, 0.5 Hz / 1 ms, 1 Hz / 2 ms. Target gains at the 50% point: about 2.8, 1.7 and 1.2 dB. `ft8sim`'s Watterson implementation may differ in detail from the simulator's; a few tenths of a dB is not a bug.
4. **Frequency robustness.** Sweep `f0` in 0.1 Hz steps across one tone spacing at a fixed SNR and confirm the coherent gain does not depend on the fractional offset. Add runs with a drifting oscillator if a test signal with drift is available (a small change to `ft8sim` can add one).
5. **Real recordings.** Run both decoders on an hour of recorded band audio and compare the decode lists: new decodes gained, old decodes lost (should be none), false decodes (should stay at the CRC floor). The multi-signal environment is the thing the simulation did not test.
6. **Estimator sanity.** Log the estimated offset, chosen Doppler hypothesis, candidate index used and number of DD passes per decode. On `ft8sim` files the offset error should be a few mHz on AWGN; on fading files tens of mHz.
7. **Time budget.** Measure wall time per 15 s cycle with a busy passband. If the coherent path does not fit, drop to three candidates and two DD passes first.

Numbers to write down for each run: SNR, files, decodes off, decodes on, false decodes, mean decode time.

## 6. Pitfalls and open items

The simulation left out four things a real decoder meets, in the order they are likely to matter.

1. **GFSK pulse shape.** FT8 smooths tone transitions with a Gaussian pulse (BT = 2.0), so the phase near each symbol boundary depends slightly on the neighbouring tones. The simulator switched tones abruptly. Correlating against pure tones over the full symbol, as the noncoherent decoder does, loses a small amount; correlating over the central 80% of each symbol, or against the true pulse given the neighbours' tentative decisions, recovers it. Measure before optimising.
2. **Time sync.** The simulation assumed the symbol boundaries are known. A residual timing error reduces the correlation peak and leaks energy into adjacent symbols. If the coherent gain comes out short of the simulation, try a small DT search (±1/4 symbol in 1/16 steps) inside the coherent path, scored by the same likelihood.
3. **Multi-signal passbands.** A strong neighbour inside the 50 Hz window biases the noise and channel estimates. Decoding strongest-first with subtraction limits this; the coherent channel estimate can improve the subtraction itself. Watch the false-decode rate on real recordings.
4. **Phase noise and drift.** Drift up to about 1 Hz per minute costs nothing; beyond that a drift term in the search (a second, coarse grid dimension) would be needed. Transmitter phase noise was not modelled; the partially coherent metric degrades toward noncoherent as the estimate worsens, so the floor is today's performance.

Not tried, could add more: per-tone channel estimation for frequency-selective paths (only matters above ~150 Hz of signal bandwidth); joint channel and data estimation by expectation-maximisation instead of the heuristic soft-observation weights in step 10 (the gap to perfect channel knowledge was still 1.7–3.1 dB at 30 s per message on fading paths); using the coherent channel estimate in `subtractft8`.

Two limits to keep in mind: the coherent gain is one-directional per link until both stations run it, and at the slowest symbol rates on disturbed paths (0.5 s coherence time) no receiver recovers much, because the phase changes faster than it can be measured.

## 7. Opening prompt for the Claude Code session

```
I want to prototype a coherent FT8 decoder in this WSJT-X checkout. The receiver
change is decode-side only; the transmitted signal stays standard FT8.

Read these first, in order:
1. tools/wsim/HANDOFF.md  (the brief: algorithm, integration points, validation plan)
2. tools/wsim/README.md and tools/wsim/wsim.py  (the reference implementation;
   the functions search_offset, wiener_smooth, FSKScheme._coherent_llr,
   FSKScheme.refine and run_point are the specification)
3. The interactive report, if you can open claude.ai links:
   https://claude.ai/artifact/MVzoWbo79iRMHCV4sJUDPH

Then, before writing any Fortran:
- Confirm the file and routine names in HANDOFF.md section 4 against this tree
  (rg for four2a, bmeta, bpdecode174_91, ft8_downsample, sync8d, subtractft8).
- Tell me where the complex per-symbol spectra are formed in ft8b and where
  bpdecode174_91 holds its posterior LLRs.
- Propose the smallest patch that adds the coherent path behind a switch, run
  only when the existing decode fails, with the noncoherent path bit-exact.

Plan: (1) expose posterior LLRs from the BP decoder; (2) new ft8_coherent.f90
implementing steps 2-11 of the handoff; (3) hook into ft8b after the existing
attempts; (4) the ft8sim/jt9 validation from section 5, starting with the AWGN
curve at -19 to -24 dB with the path off and on.
```
