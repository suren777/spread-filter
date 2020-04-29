import numpy as np
from price_generator import dn, An, B, g, R


def filter(
    yn, xnn1, xnn, pnn1, pn1n1, T1, T2, a, l, sig, q, k, c1, c2, c3, c4, dt
):
    d1, d2 = dn(a, l, sig, k, [T1, T2], c1, c2, c3, c4)
    a1, a2 = An([T1, T2], k)
    d1a1pn1 = d1 + 0.5 * a1 ** 2 * pnn1
    d2a2pn1 = d2 + 0.5 * a2 ** 2 * pnn1
    en1yn = np.exp(a1 * xnn1 + d1a1pn1) - np.exp(a2 * xnn1 + d2a2pn1)
    vn = yn - en1yn
    signn1 = (
        (
            np.exp(2 * a1 * xnn1 + 2 * d1a1pn1)
            + np.exp(2 * a2 * xnn1 + 2 * d2a2pn1)
            - 2
            * np.exp(
                (a1 + a2) * xnn1 + (d1 + d2) * 0.5 * (a1 + a2) ** 2 * pnn1
            )
        )
        + q * q
        - en1yn ** 2
    )
    b = B(k, dt)
    gs = g(a, k, dt)
    r = R(sig, k, dt)
    z1n = np.exp(a1 * (b * xnn + gs + d1))
    z2n = np.exp(a2 * (b * xnn + gs + d2))
    zdiff = (
        z1n * np.exp(0.5 * (a1 * b) ** 2 * pn1n1 + r ** 2) * a1
        - z2n * np.exp(0.5 * (a2 * b) ** 2 * pn1n1 + r ** 2) * a2
    )
    kstar = (b ** 2 * pn1n1 * zdiff) / signn1
    # need to discuss eq 41
    pnn = (
        b ** 2 * pn1n1
        + kstar ** 2 * signn1
        - 2 * b ** 2 * kstar * pn1n1 * zdiff
    )
    pn1n = b ** 2 * pnn + r ** 2
    xnn = xnn1 + kstar * vn
    xnn1 = b * xnn + gs
    return xnn1, xnn, pn1n, pnn
