"""Extract FT8's (174,91) LDPC code from ft8_lib's constants.c into a .npz file.

Usage: python3 -I parse_ft8.py <constants.c> <out.npz>
"""
import re
import sys

import numpy as np


def array_body(src, name):
    m = re.search(name + r"\s*\[[^\]]*\]\s*(\[[^\]]*\])?\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        raise SystemExit(f"array {name} not found")
    body = re.sub(r"//[^\n]*", "", m.group(2))
    return [int(t, 0) for t in re.findall(r"0x[0-9a-fA-F]+|\d+", body.replace("u", ""))]


def main(path, out):
    src = open(path).read()
    N, K, M = 174, 91, 83
    nm = np.array(array_body(src, "kFTX_LDPC_Nm"), dtype=int).reshape(M, 7)
    gen = np.array(array_body(src, "kFTX_LDPC_generator"), dtype=np.uint8).reshape(M, 12)

    H = np.zeros((M, N), dtype=np.uint8)
    for r in range(M):
        for c in nm[r]:
            if c > 0:
                H[r, c - 1] = 1
    # generator rows give the 83 parity bits as functions of the 91 message bits
    P = np.unpackbits(gen, axis=1)[:, :K]  # (83, 91)

    # Check: codeword = [msg, P@msg] must satisfy H c = 0 for random messages
    rng = np.random.default_rng(1)
    for _ in range(20):
        msg = rng.integers(0, 2, K).astype(np.uint8)
        cw = np.concatenate([msg, (P @ msg) % 2])
        assert not ((H.astype(int) @ cw) % 2).any(), "generator/parity mismatch"
    col_w = H.sum(0)
    row_w = H.sum(1)
    print(f"FT8 LDPC ok: H {H.shape}, col weights {set(col_w)}, row weights {sorted(set(row_w))}")
    np.savez(out, H=H, P=P)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
