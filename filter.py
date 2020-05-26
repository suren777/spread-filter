import numpy as np
from price_generator import dn, An, B, g, R


class Filter:
    def __init__(self, a, l, sig, q, k, c1, c2, c3, c4, dt):
        (
            self.a,
            self.l,
            self.sig,
            self.q,
            self.k,
            self.c1,
            self.c2,
            self.c3,
            self.c4,
            self.dt,
        ) = (a, l, sig, q, k, c1, c2, c3, c4, dt)

    def one_step(self, yn, xnn, pn1n1, T1, T2, T1_new, T2_new):
        T = np.array([T1, T2])
        d1, d2 = dn(
            self.a,
            self.l,
            self.sig,
            self.k,
            T,
            self.c1,
            self.c2,
            self.c3,
            self.c4,
        )
        b = B(self.k, self.dt)
        gs = g(self.a, self.k, self.dt)
        r = R(self.sig, self.k, self.dt)
        pnn1 = b ** 2 * pn1n1 + r ** 2
        xnn1 = b * (xnn) + gs
        a1, a2 = An(T, self.k)
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
            + self.q ** 2
            - en1yn ** 2
        )
        z1n = np.exp(a1 * (xnn1 + d1))
        z2n = np.exp(a2 * (xnn1 + d2))
        zdiff = (
            z1n * np.exp(0.5 * (a1 * b) ** 2 * pn1n1 + r ** 2) * a1
            - z2n * np.exp(0.5 * (a2 * b) ** 2 * pn1n1 + r ** 2) * a2
        )
        kstar = (b ** 2 * pn1n1 * zdiff) / signn1
        # need to discuss eq 41
        pnn = b ** 2 * pn1n1 + r ** 2 - (kstar ** 2) * signn1
        xnn = xnn1 + kstar * vn
        xnn1 = b * xnn + gs

        T = np.array([T1_new, T2_new])
        d1, d2 = dn(
            self.a,
            self.l,
            self.sig,
            self.k,
            T,
            self.c1,
            self.c2,
            self.c3,
            self.c4,
        )
        pnn1 = b ** 2 * pnn + r ** 2
        a1, a2 = An(T, self.k)
        d1a1pn1 = d1 + 0.5 * a1 ** 2 * pnn1
        d2a2pn1 = d2 + 0.5 * a2 ** 2 * pnn1
        en1yn = np.exp(a1 * xnn1 + d1a1pn1) - np.exp(a2 * xnn1 + d2a2pn1)
        return xnn1, pnn, en1yn
