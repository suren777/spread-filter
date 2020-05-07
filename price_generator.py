import numpy as np


def seasonality_function(t, c1, c2, c3, c4):
    return c1 + c2 * np.sin(c3 * t + c4)


def gaussian_noise():
    return 0


def R(sig, k, dt):
    return sig ** 2 * 0.5 / k * (1 - np.exp(-2 * k * dt))


def An(T, k, t0=0):
    return np.exp(-k * (T - t0))


def g(a, k, dt):
    return a / k * (1 - B(k, dt))


def B(k, dt):
    return np.exp(-k * dt)


def dn(a, l, sig, k, T, c1, c2, c3, c4):
    return (
        (a - l * sig) / k * (1 - np.exp(-k * (T)))
        + sig ** 2 * 0.25 / k * (1 - np.exp(-2 * k * (T)))
        + seasonality_function(T, c1, c2, c3, c4)
    )


def xtdt(x0, a, sig, k, dt, w):
    return B(k, dt) * x0 + g(a, k, dt) + np.sqrt(R(sig, k, dt) * dt) * w


def log_futures(spot, T, k, a, l, sig, eta, c1, c2, c3, c4, q_noise):
    return (
        An(T, k) * spot + dn(a, l, sig, k, T, c1, c2, c3, c4) + q_noise * eta
    )


def cal_spread_det(spot, T1, T2, k, a, l, sig, c1, c2, c3, c4):
    T = np.array([T1, T2])
    d1, d2 = dn(a, l, sig, k, T, c1, c2, c3, c4)
    a1, a2 = An(T, k)
    return np.exp(a1 * spot + d1) - np.exp(a2 * spot + d2)


def calendar_spread(
    spot, T1, T2, k, a, l, sig, eta, c1, c2, c3, c4, q_noise, dt
):
    return cal_spread_det(
        spot, T1, T2, k, a, l, sig, c1, c2, c3, c4
    ) + q_noise * eta * np.sqrt(dt)


def generate_futures_path(
    x0,
    a,
    l,
    sig,
    k,
    dt,
    Tn,
    c1,
    c2,
    c3,
    c4,
    eta,
    n_steps=100,
    rolling=True,
    seed=42,
):
    n_fut = len(Tn)
    np.random.seed(seed)
    wiener = np.random.normal(size=n_steps)
    q = np.random.normal(size=(n_fut, n_steps))
    spot = np.zeros(shape=n_steps + 1)
    futures = np.zeros(shape=(n_fut, n_steps))
    spot[0] = x0
    T = np.array(Tn)
    Ts = [T]
    for i, w in enumerate(wiener):
        spot[i + 1] = xtdt(spot[i], a, sig, k, dt, w)
        futures[:, i] = log_futures(
            spot[i], T, k, a, l, sig, eta, c1, c2, c3, c4, q[:, i], dt
        )
        if not rolling:
            for i, t in enumerate(list(T)):
                if T[i] - dt >= 0:
                    T[i] -= dt
                else:
                    T[i] = Tn[i] + np.random.uniform(-3 / 252.0, 3 / 252.0)
        Ts.append(T)
    return spot, futures, Ts


def generate_spread_path(
    x0,
    a,
    l,
    sig,
    k,
    dt,
    Tn,
    c1,
    c2,
    c3,
    c4,
    eta,
    n_steps=100,
    rolling=True,
    seed=42,
):
    np.random.seed(seed)
    wiener = np.random.normal(size=n_steps)
    q = np.random.normal(size=n_steps)
    spot = np.zeros(shape=n_steps)
    spreads = np.zeros(shape=n_steps)
    spot[0] = x0
    Ts = np.zeros((len(Tn), len(wiener)))
    Ts[:, 0] = Tn
    for i, w in enumerate(wiener):
        if i > 0:
            spot[i] = xtdt(spot[i - 1], a, sig, k, dt, w)
        spreads[i] = calendar_spread(
            spot[i],
            Ts[0, i],
            Ts[1, i],
            k,
            a,
            l,
            sig,
            eta,
            c1,
            c2,
            c3,
            c4,
            q[i],
            dt,
        )
        if not rolling:
            for j, t in enumerate(list(Ts[:, i])):
                try:
                    if t - dt >= 0:
                        Ts[j, i + 1] = Ts[j, i] - dt
                    else:
                        Ts[j, i + 1] = Tn[j]
                except:
                    pass
    return spot, spreads, Ts


def one_factor_model():
    pass


if __name__ == "__main__":
    x0, alpha, lmbd, sigma, kappa, dt = 30, 0.0658, 0, 0.43, 5, 1 / 252.0
    generate_futures_path(x0, alpha, lmbd, sigma, kappa, dt)
