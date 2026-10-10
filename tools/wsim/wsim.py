"""wsim: weak-signal HF mode simulator.

Compares noncoherent M-FSK and pilot-aided coherent BPSK, each protected by an
LDPC code, on AWGN and Watterson-model HF channels (ITU-R F.1487 mid-latitude
quiet / moderate / disturbed). SNR is referenced to a 2500 Hz noise bandwidth,
the convention WSJT-X uses for its reported SNR, so results are directly
comparable to FT8/FT4 decode thresholds.

Assumptions (stated so the results can be read correctly):
  * Perfect time and frequency synchronisation. Sync/pilot symbols are still
    transmitted and still cost energy and air time (same overhead fraction as
    FT8: 21 of 79 symbols), but acquisition itself is not simulated.
  * The receiver knows the noise level and the long-term average signal level
    (standard "genie SNR" assumption for comparing modulations).
  * Decoder: belief propagation (sum-product), up to 60 iterations, no OSD and
    no a-priori message knowledge. Success = every payload+CRC bit correct.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# LDPC codes
# ---------------------------------------------------------------------------


class LDPC:
    """Binary LDPC code with a systematic encoder and BP decoder structures."""

    def __init__(self, H: np.ndarray, name: str, P: np.ndarray | None = None):
        H = (np.asarray(H) % 2).astype(np.uint8)
        self.H, self.name = H, name
        self.m, self.n = H.shape
        if P is not None:  # explicit generator: codeword = [msg, P msg]
            self.k = P.shape[1]
            self.info_pos = np.arange(self.k)
            self.par_pos = np.arange(self.k, self.n)
            self.A = P.astype(np.uint8)
        else:
            self._systematic_from_H()
        self._build_graph()

    def _systematic_from_H(self):
        Hr = self.H.copy()
        m, n = Hr.shape
        pivots, row = [], 0
        for col in range(n):
            if row >= m:
                break
            hit = np.nonzero(Hr[row:, col])[0]
            if hit.size == 0:
                continue
            p = row + hit[0]
            if p != row:
                Hr[[row, p]] = Hr[[p, row]]
            others = np.nonzero(Hr[:, col])[0]
            others = others[others != row]
            Hr[others] ^= Hr[row]
            pivots.append(col)
            row += 1
        rank = row
        Hr = Hr[:rank]
        self.par_pos = np.array(pivots)
        self.info_pos = np.array([c for c in range(n) if c not in set(pivots)])
        self.k = self.info_pos.size
        self.A = Hr[:, self.info_pos]  # parity[i] = A[i] . info

    def encode(self, msg: np.ndarray) -> np.ndarray:
        msg = msg.astype(np.uint8)
        cw = np.zeros((msg.shape[0], self.n), dtype=np.uint8)
        cw[:, self.info_pos] = msg
        cw[:, self.par_pos] = (msg.astype(np.int32) @ self.A.T.astype(np.int32)) % 2
        return cw

    def _build_graph(self):
        rows, cols = np.nonzero(self.H)
        order_c = np.lexsort((cols, rows))
        self.e_chk = rows[order_c].astype(np.int64)
        self.e_var = cols[order_c].astype(np.int64)
        E = self.e_var.size
        self.chk_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.e_chk, minlength=self.m))]).astype(np.int64)
        by_var = np.argsort(self.e_var, kind="stable")
        self.var_edges = by_var.astype(np.int64)
        self.var_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.e_var, minlength=self.n))]).astype(np.int64)
        assert self.var_ptr[-1] == E

    def decode(self, llr: np.ndarray, max_iter: int = 60):
        """Returns (hard decisions (F,n) uint8, posterior LLRs (F,n))."""
        return _bp_decode(np.ascontiguousarray(llr, dtype=np.float64), self.e_var, self.chk_ptr,
                          self.var_edges, self.var_ptr, max_iter)

    def syndrome_ok(self, hard: np.ndarray) -> np.ndarray:
        return ~(((hard.astype(np.int32) @ self.H.T.astype(np.int32)) % 2).any(axis=1))


@njit(parallel=True, cache=True)
def _bp_decode(llr, e_var, chk_ptr, var_edges, var_ptr, max_iter):
    F, n = llr.shape
    m = chk_ptr.size - 1
    E = e_var.size
    out = np.zeros((F, n), dtype=np.uint8)
    post = np.zeros((F, n))
    for f in prange(F):
        v2c = np.empty(E)
        c2v = np.zeros(E)
        t = np.empty(E)
        hard = np.zeros(n, dtype=np.uint8)
        tots = np.zeros(n)
        for e in range(E):
            v2c[e] = llr[f, e_var[e]]
        for it in range(max_iter):
            # check-node update (tanh rule, leave-one-out by prefix/suffix products)
            for c in range(m):
                a, b = chk_ptr[c], chk_ptr[c + 1]
                for e in range(a, b):
                    x = v2c[e]
                    if x > 30.0:
                        x = 30.0
                    elif x < -30.0:
                        x = -30.0
                    t[e] = math.tanh(0.5 * x)
                pre = 1.0
                for e in range(a, b):
                    c2v[e] = pre
                    pre *= t[e]
                suf = 1.0
                for e in range(b - 1, a - 1, -1):
                    p = c2v[e] * suf
                    suf *= t[e]
                    if p > 0.999999999:
                        p = 0.999999999
                    elif p < -0.999999999:
                        p = -0.999999999
                    c2v[e] = 2.0 * math.atanh(p)
            # variable-node update
            for v in range(n):
                a, b = var_ptr[v], var_ptr[v + 1]
                tot = llr[f, v]
                for j in range(a, b):
                    tot += c2v[var_edges[j]]
                for j in range(a, b):
                    e = var_edges[j]
                    v2c[e] = tot - c2v[e]
                hard[v] = 1 if tot < 0.0 else 0
                tots[v] = tot
            # syndrome check
            ok = True
            for c in range(m):
                s = 0
                for e in range(chk_ptr[c], chk_ptr[c + 1]):
                    s ^= hard[e_var[e]]
                if s:
                    ok = False
                    break
            if ok:
                break
        for v in range(n):
            out[f, v] = hard[v]
            post[f, v] = tots[v]
    return out, post


def peg_ldpc(n: int, k: int, dv: int = 3, seed: int = 0, name: str | None = None) -> LDPC:
    """Progressive-edge-growth LDPC construction (Hu, Eleftheriou, Arnold 2005)."""
    rng = np.random.default_rng(seed)
    m = n - k
    chk_deg = np.zeros(m, dtype=int)
    var_nbrs = [[] for _ in range(n)]
    chk_nbrs = [[] for _ in range(m)]
    for j in range(n):
        for d in range(dv):
            if d == 0:
                cand = np.flatnonzero(chk_deg == chk_deg.min())
            else:
                # BFS from variable j through the current graph
                seen_c = set(var_nbrs[j])
                frontier_c = set(var_nbrs[j])
                seen_v = {j}
                while True:
                    nv = set()
                    for c in frontier_c:
                        nv.update(chk_nbrs[c])
                    nv -= seen_v
                    seen_v |= nv
                    nc = set()
                    for v in nv:
                        nc.update(var_nbrs[v])
                    nc -= seen_c
                    if not nc or len(seen_c | nc) == m:
                        if len(seen_c | nc) == m and nc:
                            # all checks reachable: pick from the last new layer
                            cand_set = nc
                        else:
                            cand_set = set(range(m)) - seen_c
                        break
                    seen_c |= nc
                    frontier_c = nc
                cand = np.array(sorted(cand_set - set(var_nbrs[j])))
                if cand.size == 0:
                    cand = np.array([c for c in range(m) if c not in var_nbrs[j]])
                cand = cand[chk_deg[cand] == chk_deg[cand].min()]
            c = int(rng.choice(cand))
            var_nbrs[j].append(c)
            chk_nbrs[c].append(j)
            chk_deg[c] += 1
    H = np.zeros((m, n), dtype=np.uint8)
    for j in range(n):
        H[var_nbrs[j], j] = 1
    return LDPC(H, name or f"PEG({n},{k})")


def ft8_code(path: str) -> LDPC:
    d = np.load(path)
    return LDPC(d["H"], "FT8 LDPC(174,91)", P=d["P"])


# ---------------------------------------------------------------------------
# HF channel (Watterson / ITU-R F.1487)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    name: str
    delay_s: float = 0.0  # differential delay between the two paths
    doppler_hz: float = 0.0  # frequency spread, 2*sigma of Gaussian Doppler spectrum

    @property
    def fading(self) -> bool:
        return self.doppler_hz > 0


CHANNELS = {
    "awgn": Channel("AWGN (no fading)"),
    "quiet": Channel("ITU mid-lat quiet: 0.5 ms, 0.1 Hz", 0.5e-3, 0.1),
    "moderate": Channel("ITU mid-lat moderate: 1 ms, 0.5 Hz", 1.0e-3, 0.5),
    "disturbed": Channel("ITU mid-lat disturbed: 2 ms, 1 Hz", 2.0e-3, 1.0),
}


def fading_taps(rng, F: int, nsamp: int, fs: float, doppler: float) -> np.ndarray:
    """Two independent Rayleigh paths (power 1/2 each), Gaussian Doppler spectrum.

    Returns complex array (2, F, nsamp)."""
    sigma = doppler / 2.0
    ff = max(10.0 * doppler, 2.0)  # generation rate (Hz), >> spectrum width
    T = nsamp / fs
    L = 1 << int(math.ceil(math.log2(max(ff * T * 2 + 8, ff * 40.0 / doppler))))
    w = (rng.standard_normal((2, F, L)) + 1j * rng.standard_normal((2, F, L)))
    f = np.fft.fftfreq(L, 1.0 / ff)
    shape = np.exp(-(f ** 2) / (4.0 * sigma ** 2))  # sqrt of Gaussian PSD
    shape /= math.sqrt(np.sum(shape ** 2) / L)
    g = np.fft.ifft(np.fft.fft(w, axis=2) * shape, axis=2) * math.sqrt(0.5 / 2.0)
    # g now has variance 0.5 per path; take a segment and interpolate to fs
    nf = int(math.ceil(T * ff)) + 2
    tf = np.arange(nf) / ff
    ts = np.arange(nsamp) / fs
    idx = np.minimum((ts * ff).astype(int), nf - 2)
    frac = ts * ff - idx
    seg = g[:, :, :nf]
    return seg[:, :, idx] * (1 - frac) + seg[:, :, idx + 1] * frac


def apply_channel(rng, s: np.ndarray, fs: float, ch: Channel, snr_db: float, offset=None, drift=None):
    """s: (F, nsamp) unit-power complex baseband. Returns (r, sigma2, gain)
    where gain is the per-sample complex gain seen by a tone at 0 Hz (path sum).
    offset: (F,) frequency offset in Hz; drift: (F,) linear frequency change in Hz over the message."""
    F, nsamp = s.shape
    if offset is not None or drift is not None:
        tt = np.arange(nsamp) / fs
        T = nsamp / fs
        ph = np.zeros((F, nsamp))
        if offset is not None:
            ph += offset[:, None] * tt[None, :]
        if drift is not None:
            ph += drift[:, None] * (tt[None, :] ** 2 / (2 * T) - tt[None, :] / 2)
        s = s * np.exp(2j * np.pi * ph)
    if ch.fading:
        g = fading_taps(rng, F, nsamp, fs, ch.doppler_hz)
        guard = int(math.ceil(ch.delay_s * fs)) + 8
        npad = nsamp + guard
        S = np.fft.fft(np.pad(s, ((0, 0), (0, guard))), axis=1)
        fr = np.fft.fftfreq(npad, 1.0 / fs)
        sd = np.fft.ifft(S * np.exp(-2j * np.pi * fr * ch.delay_s), axis=1)[:, :nsamp]
        r = g[0] * s + g[1] * sd
    else:
        g = None
        r = s.copy()
    sigma2 = fs / (10 ** (snr_db / 10) * 2500.0)  # complex noise variance per sample
    r = r + math.sqrt(sigma2 / 2) * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
    return r, sigma2, g


# ---------------------------------------------------------------------------
# Channel estimation (coherent receivers)
# ---------------------------------------------------------------------------

BIG = 1e6  # noise variance assigned to "no observation"
DOPPLER_HYPS = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0])


def gauss_corr(dt: np.ndarray, doppler: float) -> np.ndarray:
    """Channel autocorrelation for a Gaussian Doppler spectrum with 2-sigma width `doppler`."""
    if doppler <= 0:
        return np.ones_like(dt, dtype=float)
    sg = doppler / 2.0
    return np.exp(-2.0 * np.pi ** 2 * sg ** 2 * dt ** 2)


def select_doppler(z, v, t, hyps=DOPPLER_HYPS, max_obs=100):
    """Pick, per frame, the Doppler hypothesis with the largest marginal likelihood of the
    observations z (F,P) with noise variances v (P,) at times t (P,)."""
    F, P = z.shape
    if P > max_obs:
        sel = np.round(np.linspace(0, P - 1, max_obs)).astype(int)
        z, v, t = z[:, sel], v[sel], t[sel]
        P = max_obs
    dt = t[:, None] - t[None, :]
    ll = np.empty((F, hyps.size))
    for i, d in enumerate(hyps):
        A = gauss_corr(dt, d) + np.diag(v)
        x = np.linalg.solve(A, z.T).T  # (F,P)
        _, logdet = np.linalg.slogdet(A)
        ll[:, i] = -np.real(np.sum(np.conj(z) * x, axis=1)) - logdet
    return hyps[np.argmax(ll, axis=1)]


def _gp_inv(t, v, d):
    A = gauss_corr(t[:, None] - t[None, :], d) + np.diag(v)
    return np.linalg.inv(A), np.linalg.slogdet(A)[1]


def search_offset(z, v, t, hyps=DOPPLER_HYPS, span=3.125, step=None, fine_span=None, fine_step=None,
                  n_cand=5, min_sep=None):
    """Joint maximum-likelihood search of frequency offset and Doppler hypothesis.

    z: (F,P) observations z_i = h(t_i) e^{j 2 pi df t_i} + noise, v: (P,) noise variances,
    t: (P,) times. The channel h is a unit-variance Gaussian process with the Watterson
    autocorrelation. Returns (cands (F, n_cand) offsets in Hz, best first, NaN where there is
    no further candidate; doppler (F,))."""
    F, P = z.shape
    T = float(t[-1] - t[0])
    # the likelihood's main lobe is ~1/T wide; FT8's three sync blocks put grating lobes at +-2/T
    if step is None:
        step = min(0.025, 0.2 / T)
    if fine_span is None:
        fine_span = 1.5 * step
    if fine_step is None:
        fine_step = step / 12
    if min_sep is None:
        min_sep = 1.0 / T
    dt = t[:, None] - t[None, :]
    W0 = np.conj(z)[:, :, None] * z[:, None, :]
    grid = np.arange(-span, span + 1e-9, step)
    G = grid.size
    Ef = np.exp(2j * np.pi * grid[:, None, None] * dt[None]).reshape(G, -1)
    full = np.asarray(hyps)
    hyps = full[::2] if full.size > 5 else full  # coarse joint search on a thinned bank
    ll_all = np.empty((F, len(hyps), G))
    for hi, d in enumerate(hyps):
        M, logdet = _gp_inv(t, v, d)
        ll_all[:, hi, :] = -np.real((W0 * M[None]).reshape(F, -1) @ Ef.T) - logdet
    flat = ll_all.reshape(F, -1)
    k = np.argmax(flat, axis=1)
    hbest, jbest = np.divmod(k, G)
    best_d = np.asarray(hyps)[hbest]
    # candidate list from the chosen hypothesis' likelihood curve: local maxima, separated
    llc = ll_all[np.arange(F), hbest, :]  # (F, G)
    cands = np.full((F, n_cand), np.nan)
    cands[:, 0] = grid[jbest]
    sep = max(1, int(round(min_sep / step)))
    inner = llc[:, 1:-1]
    is_max = (inner >= llc[:, :-2]) & (inner >= llc[:, 2:])
    for f in range(F):
        peaks = np.flatnonzero(is_max[f]) + 1
        peaks = peaks[np.argsort(-llc[f, peaks])]
        chosen = [jbest[f]]
        for p in peaks:
            if len(chosen) >= n_cand:
                break
            if all(abs(p - c) > sep for c in chosen):
                chosen.append(p)
        cands[f, : len(chosen)] = grid[chosen]
    # fine stage for every candidate
    fine = np.arange(-fine_span, fine_span + 1e-9, fine_step)
    Efine = np.exp(2j * np.pi * fine[:, None, None] * dt[None]).reshape(fine.size, -1)
    for d in np.unique(best_d):
        idx = np.flatnonzero(best_d == d)
        M, _ = _gp_inv(t, v, d)
        WM = W0[idx] * M[None]
        for c in range(n_cand):
            ok = ~np.isnan(cands[idx, c])
            if not ok.any():
                continue
            ii = idx[ok]
            Estar = np.exp(2j * np.pi * cands[ii, c][:, None, None] * dt[None])
            ll = -np.real((WM[ok] * Estar).reshape(ii.size, -1) @ Efine.T)
            cands[ii, c] = cands[ii, c] + fine[np.argmax(ll, axis=1)]
    # final Doppler choice over the full bank at the chosen offset
    zd = z * np.exp(-2j * np.pi * cands[:, :1] * t[None, :])
    best_d = select_doppler(zd, v, t, hyps=full)
    return cands, best_d


def refine_offset(z, v, t, dop, df0, span=0.3, step=0.005, fine_span=0.01, fine_step=0.001, max_obs=128):
    """Local offset refinement with per-frame observation weights (decision-directed pass).

    z, v: (F,N) observations (already derotated by the current estimate df0) and noise
    variances (BIG = unobserved), t: (N,), dop: (F,) Doppler per frame, df0: (F,) current
    offset estimate. Returns refined df (F,)."""
    F, N = z.shape
    if N > max_obs:
        sel = np.round(np.linspace(0, N - 1, max_obs)).astype(int)
        # keep the best-observed symbols: replace selection by those with smallest v on average
        order = np.argsort(np.mean(v, axis=0))[:max_obs]
        sel = np.sort(order)
        z, v, t = z[:, sel], v[:, sel], t[sel]
        N = max_obs
    dt = t[:, None] - t[None, :]
    grid = np.arange(-span, span + 1e-9, step)
    Eg = np.exp(2j * np.pi * grid[:, None, None] * dt[None]).reshape(grid.size, -1)
    fine = np.arange(-fine_span, fine_span + 1e-9, fine_step)
    Ef = np.exp(2j * np.pi * fine[:, None, None] * dt[None]).reshape(fine.size, -1)
    out = df0.copy()
    R0 = gauss_corr(dt, 0.0)
    for d in np.unique(dop):
        idx = np.flatnonzero(dop == d)
        R = gauss_corr(dt, d)
        for s0 in range(0, idx.size, 64):
            fi = idx[s0: s0 + 64]
            A = R[None] + np.einsum("fi,ij->fij", v[fi], np.eye(N))
            M = np.linalg.inv(A)
            # z is already derotated by df0, so only the residual offset is modelled here
            W = np.conj(z[fi])[:, :, None] * z[fi][:, None, :] * M
            Wf = W.reshape(fi.size, -1)
            ll = -np.real(Wf @ Eg.T)
            j = np.argmax(ll, axis=1)
            d1 = grid[j]
            E1 = np.exp(2j * np.pi * d1[:, None, None] * dt[None])
            ll2 = -np.real((W * E1).reshape(fi.size, -1) @ Ef.T)
            out[fi] = df0[fi] + d1 + fine[np.argmax(ll2, axis=1)]
    return out


def wiener_smooth(z, v, t, doppler, block=160, overlap=32):
    """MMSE smoothing of a unit-variance Gaussian-process channel.

    z: (F,N) complex observations z_i = h(t_i) + noise, v: (F,N) noise variances (BIG = none),
    t: (N,) times, doppler: (F,) per-frame Doppler spread (0 = static). Returns (hhat, errvar),
    both (F,N). Processed in overlapping time blocks so cost stays linear in N."""
    F, N = z.shape
    hhat = np.empty_like(z)
    ev = np.empty((F, N))
    if N <= block:
        spans = [(0, N, 0, N)]
    else:
        core = block - 2 * overlap
        spans = []
        c0 = 0
        while c0 < N:
            c1 = min(N, c0 + core)
            a = max(0, c0 - overlap)
            b = min(N, c1 + overlap)
            if b - a < block:
                if a == 0:
                    b = min(N, block)
                else:
                    a = max(0, b - block)
            spans.append((a, b, c0, c1))
            c0 = c1
    for d in np.unique(doppler):
        idx = np.flatnonzero(doppler == d)
        for a, b, c0, c1 in spans:
            tt = t[a:b]
            R = gauss_corr(tt[:, None] - tt[None, :], d)
            vb = v[idx, a:b]
            if np.all(vb == vb[0]):  # same weights for every frame (pilot-only pass): one solve
                G = np.linalg.solve(R + np.diag(vb[0]), R)
                h = z[idx, a:b] @ G
                e = 1.0 - np.einsum("ij,ji->i", R, G)
                hhat[idx, c0:c1] = h[:, c0 - a: c1 - a]
                ev[idx, c0:c1] = np.maximum(e[c0 - a: c1 - a], 1e-9)[None, :]
                continue
            for s0 in range(0, idx.size, 128):
                fi = idx[s0: s0 + 128]
                A = R[None, :, :] + np.einsum("fi,ij->fij", v[fi, a:b], np.eye(b - a))
                G = np.linalg.solve(A, np.broadcast_to(R, (fi.size, b - a, b - a)))  # (R+V)^-1 R
                h = np.einsum("fi,fij->fj", z[fi, a:b], G)  # z^T (R+V)^-1 R  (R symmetric)
                e = 1.0 - np.einsum("ij,fji->fi", R, G)
                hhat[fi, c0:c1] = h[:, c0 - a: c1 - a]
                ev[fi, c0:c1] = np.maximum(e[:, c0 - a: c1 - a], 1e-9)
    return hhat, ev


# ---------------------------------------------------------------------------
# Modulation schemes
# ---------------------------------------------------------------------------

SYNC_FRACTION = 21 / 79  # FT8's share of symbols spent on Costas sync


def sync_count(ndata: int, frac: float = SYNC_FRACTION) -> int:
    return int(round(ndata * frac / (1 - frac)))


class RxState:
    """Per-call receiver state carried between decision-directed passes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


@dataclass
class FSKScheme:
    """Orthogonal M-FSK (tone spacing = baud, continuous phase).

    coherent=False: classic noncoherent detection (|Y|^2 metric).
    coherent=True:  the sync tones are used as pilots for an MMSE channel estimate and the
                    tones are detected with a partially coherent metric; dd_iters > 0 adds
                    decision-directed re-estimation using the decoder's soft output."""

    M: int
    code: LDPC
    T: float  # transmission time (s)
    metric: str = "auto"  # 'bessel' | 'rayleigh' | 'auto'   (noncoherent only)
    coherent: bool = False
    dd_iters: int = 0
    doppler: str = "auto"  # 'auto' (hypothesis test on pilots) | 'oracle'
    sync_frac: float = SYNC_FRACTION
    freq: str = "none"  # 'none' (assume zero offset) | 'search' | 'oracle' (true offset known)
    kind: str = "fsk"

    def __post_init__(self):
        self.bps = int(math.log2(self.M))
        self.ndata = int(math.ceil(self.code.n / self.bps))
        self.npad = self.ndata * self.bps - self.code.n
        self.nsync = sync_count(self.ndata, self.sync_frac)
        self.nsym = self.ndata + self.nsync
        self.baud = self.nsym / self.T
        self.Ns = max(2 * self.M, 8)
        self.fs = self.Ns * self.baud
        self.bandwidth = self.M * self.baud
        # FT8-style layout: sync in three blocks (start, middle, end)
        b = [self.nsync // 3] * 3
        for i in range(self.nsync - sum(b)):
            b[i] += 1
        mid = b[0] + self.ndata // 2
        is_sync = np.zeros(self.nsym, bool)
        is_sync[: b[0]] = True
        is_sync[mid: mid + b[1]] = True
        is_sync[self.nsym - b[2]:] = True
        self.is_sync = is_sync
        rng = np.random.default_rng(1234 + self.M)
        self.sync_tones = rng.integers(0, self.M, self.nsync)
        self.labels = ((np.arange(self.M)[:, None] >> np.arange(self.bps)[::-1]) & 1).astype(np.uint8)
        self.t = (np.arange(self.nsym) + 0.5) / self.baud
        self.name = f"{self.M}-FSK" + (" coherent" + (f" +DD{self.dd_iters}" if self.dd_iters else "") if self.coherent else "")
        if self.freq != "none":
            self.name += f" [freq {self.freq}]"
        self.t_samp = np.arange(self.nsym * self.Ns) / self.fs

    def modulate(self, cw: np.ndarray) -> np.ndarray:
        F = cw.shape[0]
        bits = np.concatenate([cw, np.zeros((F, self.npad), np.uint8)], axis=1).reshape(F, self.ndata, self.bps)
        tones_d = (bits * (1 << np.arange(self.bps)[::-1])).sum(-1)
        tones = np.empty((F, self.nsym), dtype=np.int64)
        tones[:, self.is_sync] = self.sync_tones
        tones[:, ~self.is_sync] = tones_d
        n = np.arange(self.Ns)
        wave = np.exp(2j * np.pi * tones[:, :, None] * n[None, None, :] / self.Ns)
        return wave.reshape(F, self.nsym * self.Ns)

    def _bit_llrs(self, m):
        """m: (F, ndata, M) tone log-metrics -> (F, n) bit LLRs (positive = 0)."""
        if self.npad:
            allowed = (self.labels[:, self.bps - self.npad:] == 0).all(1)
            m = m.copy()
            m[:, -1, ~allowed] = -np.inf
        F = m.shape[0]
        L = np.empty((F, self.ndata, self.bps))
        for b in range(self.bps):
            z = self.labels[:, b] == 0
            L[:, :, b] = _lse(m[:, :, z]) - _lse(m[:, :, ~z])
        return L.reshape(F, -1)[:, : self.code.n]

    def bins(self, r, df):
        """Tone correlations per symbol at frequency offset df (F,) Hz: (F, nsym, M)."""
        F = r.shape[0]
        rr = r * np.exp(-2j * np.pi * df[:, None] * self.t_samp[None, :])
        return np.fft.fft(rr.reshape(F, self.nsym, self.Ns), axis=2)[:, :, : self.M] / self.Ns

    def _sync_obs(self, Y):
        return Y[:, self.is_sync, :][:, np.arange(self.nsync), self.sync_tones]

    def receive(self, r, sigma2, ch: Channel, gain=None, offset=None) -> RxState:
        F = r.shape[0]
        sb2 = sigma2 / self.Ns
        df = np.zeros(F)
        if self.freq == "oracle" and offset is not None:
            df = offset.copy()
        Y = self.bins(r, df)
        st = RxState(Y=Y, sb2=sb2, r=r, df=df)
        if not self.coherent:
            if self.freq == "search":
                # noncoherent fine sync: maximise sync-tone energy over a coarse offset grid
                grid = np.arange(-3.0, 3.01, 0.25)
                best = np.full(F, -np.inf)
                for g in grid:
                    e = np.sum(np.abs(self._sync_obs(self.bins(r, np.full(F, g)))) ** 2, axis=1)
                    upd = e > best
                    best[upd] = e[upd]
                    df[upd] = g
                Y = self.bins(r, df)
                st.Y, st.df = Y, df
            Yd = Y[:, ~self.is_sync, :]
            E = np.abs(Yd) ** 2
            metric = self.metric
            if metric == "auto":
                metric = "rayleigh" if ch.fading else "bessel"
            if metric == "bessel":
                x = 2.0 * np.sqrt(E) / sb2
                m = np.where(x < 50, np.log(np.i0(np.minimum(x, 50))), x - 0.5 * np.log(2 * np.pi * np.maximum(x, 1e-9)))
            else:
                snr_sym = 1.0 / sb2
                m = E / sb2 * (snr_sym / (1.0 + snr_sym))
            st.llr = self._bit_llrs(m)
            return st
        # coherent: sync tones are pilots
        ts = self.t[self.is_sync]
        vs = np.full(self.nsync, sb2)
        if self.freq == "search":
            cands, dop = search_offset(self._sync_obs(Y), vs, ts)
            df = cands[:, 0].copy()
            Y = self.bins(r, df)
            st.Y, st.df, st.cands = Y, df, cands
            if self.doppler == "oracle":
                dop = np.full(F, ch.doppler_hz)
        elif self.doppler == "oracle":
            dop = np.full(F, ch.doppler_hz)
        else:
            dop = select_doppler(self._sync_obs(Y), vs, ts)
        st.dop = dop
        st.llr = self._pilot_llr(st, np.arange(F))
        return st

    def _pilot_llr(self, st, idx):
        """Pilot-only (sync tones) coherent LLRs for frames idx at their current offset."""
        F = idx.size
        Y = st.Y[idx]
        z = np.zeros((F, self.nsym), complex)
        v = np.full((F, self.nsym), BIG)
        z[:, self.is_sync] = self._sync_obs(Y)
        v[:, self.is_sync] = st.sb2
        sub = RxState(Y=Y, sb2=st.sb2, dop=st.dop[idx])
        return self._coherent_llr(sub, z, v)

    def retry(self, st, idx, df):
        """Restart frames idx at a different offset candidate df (len(idx),)."""
        st.df[idx] = df
        st.Y[idx] = self.bins(st.r[idx], df)
        return self._pilot_llr(st, idx)

    def _coherent_llr(self, st, z, v):
        hh, ev = wiener_smooth(z, v, self.t, st.dop)
        Yd = st.Y[:, ~self.is_sync, :]
        hd = hh[:, ~self.is_sync][:, :, None]
        evd = ev[:, ~self.is_sync][:, :, None]
        sb2 = st.sb2
        tot = sb2 + evd
        m = 2.0 * np.real(np.conj(hd) * Yd) / tot + np.abs(Yd) ** 2 * (1.0 / sb2 - 1.0 / tot)
        return self._bit_llrs(m)

    def refine(self, st, post, idx):
        """Decision-directed pass for frames idx using posterior bit LLRs post (len(idx), n)."""
        F = idx.size
        Lb = np.concatenate([post, np.full((F, self.npad), 40.0)], axis=1).reshape(F, self.ndata, self.bps)
        # log P(tone k) = sum_b log P(bit_b = label_kb)
        logp = np.zeros((F, self.ndata, self.M))
        for b in range(self.bps):
            lab = self.labels[:, b].astype(float)  # 0/1
            # log sigma(L) for bit=0, log sigma(-L) for bit=1
            L = Lb[:, :, b][:, :, None]
            logp += -np.logaddexp(0.0, -(1 - 2 * lab)[None, None, :] * L)
        logp -= _lse(logp)[..., None]
        p = np.exp(logp)
        q = np.sum(p ** 2, axis=2)
        q3 = np.sum(p ** 3, axis=2)
        good = q > 1e-3
        qq = np.where(good, q, 1.0)
        vd = np.where(good, (st.sb2 * qq + np.maximum(q3 - qq ** 2, 0.0)) / qq ** 2, BIG)

        def soft_obs(Y):
            Yd = Y[:, ~self.is_sync, :]
            zsoft = np.sum(p * Yd, axis=2)
            z = np.zeros((F, self.nsym), complex)
            v = np.full((F, self.nsym), BIG)
            z[:, ~self.is_sync] = np.where(good, zsoft / qq, 0.0)
            v[:, ~self.is_sync] = np.minimum(vd, BIG)
            z[:, self.is_sync] = self._sync_obs(Y)
            v[:, self.is_sync] = st.sb2
            return z, v

        Y = st.Y[idx]
        z, v = soft_obs(Y)
        if self.freq == "search":
            df_new = refine_offset(z, v, self.t, st.dop[idx], st.df[idx])
            Y = self.bins(st.r[idx], df_new)
            st.Y[idx], st.df[idx] = Y, df_new
            z, v = soft_obs(Y)
        sub = RxState(Y=Y, sb2=st.sb2, dop=st.dop[idx])
        return self._coherent_llr(sub, z, v)


def _lse(a):
    mx = np.max(a, axis=-1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return (mx + np.log(np.sum(np.exp(a - mx), axis=-1, keepdims=True)))[..., 0]


@dataclass
class BPSKScheme:
    """Coherent BPSK with interleaved pilot symbols.

    est='genie'  : true channel (upper bound)
    est='pilot'  : boxcar average of pilots over `window` seconds (None = whole frame)
    est='wiener' : MMSE interpolation with Doppler spread chosen per frame ('auto') or known
                   ('oracle'); dd_iters > 0 adds decision-directed re-estimation."""

    code: LDPC
    T: float
    est: str = "pilot"
    window: float | None = None
    pilot_frac: float = SYNC_FRACTION
    dd_iters: int = 0
    doppler: str = "auto"
    freq: str = "none"  # 'none' | 'search' | 'oracle'
    kind: str = "bpsk"

    def __post_init__(self):
        self.ndata = self.code.n
        self.nsync = sync_count(self.ndata, self.pilot_frac)
        self.nsym = self.ndata + self.nsync
        self.baud = self.nsym / self.T
        self.Ns = 4
        self.fs = self.Ns * self.baud
        self.bandwidth = 2 * self.baud  # null-to-null, rectangular pulses
        piv = np.unique(np.round(np.linspace(0, self.nsym - 1, self.nsync)).astype(int))
        is_p = np.zeros(self.nsym, bool)
        is_p[piv] = True
        self.is_sync = is_p
        self.nsync = int(is_p.sum())
        self.ndata = self.nsym - self.nsync
        assert self.ndata == self.code.n
        self.t = (np.arange(self.nsym) + 0.5) / self.baud
        self.t_p, self.t_d = self.t[is_p], self.t[~is_p]
        tag = {"genie": "genie channel", "pilot": "pilot-aided", "wiener": "MMSE"}[self.est]
        if self.dd_iters:
            tag += f" +DD{self.dd_iters}"
        if abs(self.pilot_frac - SYNC_FRACTION) > 1e-6:
            tag += f", {self.pilot_frac:.0%} pilots"
        if self.freq != "none":
            tag += f" [freq {self.freq}]"
        self.name = f"BPSK coherent ({tag})"
        self.t_samp = np.arange(self.nsym * self.Ns) / self.fs
        self.set_window(self.window)

    def set_window(self, window):
        self.window = window
        if self.est == "pilot":
            if window is None:
                W = np.ones((self.ndata, self.nsync))
            else:
                W = (np.abs(self.t_d[:, None] - self.t_p[None, :]) <= window / 2).astype(float)
                empty = W.sum(1) == 0
                if empty.any():
                    near = np.argmin(np.abs(self.t_d[empty, None] - self.t_p[None, :]), axis=1)
                    W[np.flatnonzero(empty), near] = 1.0
            self.npil = W.sum(1)
            self.Wn = W / self.npil[:, None]

    def modulate(self, cw):
        F = cw.shape[0]
        x = np.ones((F, self.nsym))
        x[:, ~self.is_sync] = 1.0 - 2.0 * cw
        return np.repeat(x, self.Ns, axis=1).astype(complex)

    def symbols(self, r, df):
        F = r.shape[0]
        rr = r * np.exp(-2j * np.pi * df[:, None] * self.t_samp[None, :])
        return rr.reshape(F, self.nsym, self.Ns).mean(axis=2)

    def receive(self, r, sigma2, ch: Channel, gain=None, offset=None) -> RxState:
        F = r.shape[0]
        sb2 = sigma2 / self.Ns
        df = np.zeros(F)
        if self.freq == "oracle" and offset is not None:
            df = offset.copy()
        y = self.symbols(r, df)
        st = RxState(y=y, sb2=sb2, r=r, df=df)
        yd = y[:, ~self.is_sync]
        if self.est == "genie":
            if gain is None:
                h = np.ones_like(yd)
            else:
                h = (gain[0] + gain[1]).reshape(F, self.nsym, self.Ns).mean(axis=2)[:, ~self.is_sync]
            st.llr = 4.0 * np.real(np.conj(h) * yd) / sb2
            return st
        if self.est == "pilot":
            h = y[:, self.is_sync] @ self.Wn.T
            se2 = sb2 / self.npil
            st.llr = 4.0 * np.real(np.conj(h) * yd) / (sb2 + se2[None, :])
            return st
        vs = np.full(self.nsync, sb2)
        if self.freq == "search":
            cands, dop = search_offset(y[:, self.is_sync], vs, self.t_p, n_cand=2)
            df = cands[:, 0].copy()
            y = self.symbols(r, df)
            st.y, st.df, st.cands = y, df, cands
            if self.doppler == "oracle":
                dop = np.full(F, ch.doppler_hz)
        elif self.doppler == "oracle":
            dop = np.full(F, ch.doppler_hz)
        else:
            dop = select_doppler(y[:, self.is_sync], vs, self.t_p)
        st.dop = dop
        st.llr = self._pilot_llr(st, np.arange(F))
        return st

    def _pilot_llr(self, st, idx):
        y = st.y[idx]
        z = np.where(self.is_sync[None, :], y, 0.0)
        v = np.where(self.is_sync[None, :], st.sb2, BIG) * np.ones((idx.size, 1))
        sub = RxState(y=y, sb2=st.sb2, dop=st.dop[idx])
        return self._coherent_llr(sub, z, v)

    def retry(self, st, idx, df):
        st.df[idx] = df
        st.y[idx] = self.symbols(st.r[idx], df)
        return self._pilot_llr(st, idx)

    def _coherent_llr(self, st, z, v):
        hh, ev = wiener_smooth(z, v, self.t, st.dop)
        yd = st.y[:, ~self.is_sync]
        return 4.0 * np.real(np.conj(hh[:, ~self.is_sync]) * yd) / (st.sb2 + ev[:, ~self.is_sync])

    def refine(self, st, post, idx):
        F = idx.size
        y = st.y[idx]
        xh = np.ones((F, self.nsym))
        xh[:, ~self.is_sync] = np.tanh(0.5 * np.clip(post, -40, 40))
        q = xh ** 2
        good = q > 1e-3
        qq = np.where(good, q, 1.0)
        v = np.minimum(np.where(good, (st.sb2 + 1.0 - qq) / qq, BIG), BIG)
        z = np.where(good, y * xh / qq, 0.0)
        if self.freq == "search":
            df_new = refine_offset(z, v, self.t, st.dop[idx], st.df[idx])
            y = self.symbols(st.r[idx], df_new)
            st.y[idx], st.df[idx] = y, df_new
            z = np.where(good, y * xh / qq, 0.0)
        sub = RxState(y=y, sb2=st.sb2, dop=st.dop[idx])
        return self._coherent_llr(sub, z, v)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_point(scheme, ch: Channel, snr_db: float, F: int, rng, offset: float = 0.0, drift: float = 0.0,
              return_state: bool = False):
    """Simulate F frames at one SNR; return number decoded correctly.

    offset: each message gets a frequency offset uniform in +-offset Hz;
    drift: each message gets a linear frequency drift uniform in +-drift Hz over its duration."""
    code = scheme.code
    msg = rng.integers(0, 2, (F, code.k), dtype=np.uint8)
    cw = code.encode(msg)
    s = scheme.modulate(cw)
    off = rng.uniform(-offset, offset, F) if offset else None
    dr = rng.uniform(-drift, drift, F) if drift else None
    r, sigma2, g = apply_channel(rng, s, scheme.fs, ch, snr_db, off, dr)
    st = scheme.receive(r, sigma2, ch, g, offset=off)
    st.true_offset = off
    iters = getattr(scheme, "dd_iters", 0)

    def decode_with_dd(llr, idx):
        d, p = code.decode(llr)
        oks = code.syndrome_ok(d)
        for _ in range(iters):
            sub = np.flatnonzero(~oks)
            if sub.size == 0:
                break
            llr2 = scheme.refine(st, p[sub], idx[sub])
            d2, p2 = code.decode(llr2)
            d[sub], p[sub] = d2, p2
            oks[sub] = code.syndrome_ok(d2)
        return d, p, oks

    dec, post, ok_s = decode_with_dd(st.llr, np.arange(F))
    cands = getattr(st, "cands", None)
    if cands is not None:
        for c in range(1, cands.shape[1]):
            idx = np.flatnonzero(~ok_s & ~np.isnan(cands[:, c]))
            if idx.size == 0:
                break
            llr_c = scheme.retry(st, idx, cands[idx, c])
            d2, p2, ok2 = decode_with_dd(llr_c, idx)
            dec[idx], post[idx], ok_s[idx] = d2, p2, ok2
    nok = int(np.all(dec[:, code.info_pos] == msg, axis=1).sum())
    if return_state:
        return nok, st
    return nok


def thresholds(snrs, probs, levels=(0.5, 0.9)):
    """Interpolate the SNR at which decode probability crosses each level."""
    snrs, probs = np.asarray(snrs, float), np.asarray(probs, float)
    o = np.argsort(snrs)
    snrs, probs = snrs[o], np.maximum.accumulate(probs[o])  # enforce monotone
    out = {}
    for lv in levels:
        idx = np.flatnonzero(probs >= lv)
        if idx.size == 0 or idx[0] == 0:
            out[lv] = None if idx.size == 0 else float(snrs[0])
            continue
        i = idx[0]
        p0, p1 = probs[i - 1], probs[i]
        s0, s1 = snrs[i - 1], snrs[i]
        out[lv] = float(s0 + (lv - p0) / max(p1 - p0, 1e-9) * (s1 - s0))
    return out


def sweep(scheme, ch: Channel, payload_bits: int, F: int = 600, step: float | None = None,
          seed: int = 0, ebn0_guess: float = 6.0, pmax: float = 0.97, max_pts: int = 30,
          offset: float = 0.0, drift: float = 0.0):
    """Adaptive SNR sweep from below threshold until decode probability >= pmax."""
    rng = np.random.default_rng(seed)

    def point(snr):
        return run_point(scheme, ch, snr, F, rng, offset=offset, drift=drift) / F
    R = payload_bits / scheme.T
    if step is None:
        step = 1.0 if ch.fading else 0.5
    snr = round((ebn0_guess - 10 * math.log10(2500 / R)) / step) * step
    pts = {}
    # walk down until probability small
    while len(pts) < max_pts:
        p = point(snr)
        pts[snr] = p
        if p <= 0.05:
            break
        snr -= step
    snr = max(pts) + step
    while len(pts) < max_pts and max(pts.values()) < pmax:
        if snr not in pts:
            pts[snr] = point(snr)
        snr += step
    s = sorted(pts)
    p = [pts[x] for x in s]
    return s, p, thresholds(s, p)
