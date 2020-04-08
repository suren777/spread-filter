import numpy as np


def seasonality_function(t, c1, c2, c3, c4):
    return c1 + c2 * np.sin(c3 * t + c4)


def gaussian_noise():
    return 0


def R(sig, k, dt):
    return sig * sig / 2.0 / k * (1 - np.exp(-2 * k * dt))


def An(T, k, t0=0):
    return np.exp(-k * (T - t0))


def g(a, k, dt):
    return a / k * (1 - np.exp(-k * dt))


def B(k, dt):
    return np.exp(-k * dt)


def dn(a, l, sig, k, T, c1, c2, c3, c4, t0=0):
    return (a - l * sig) / k * (
        1 - np.exp(-k * (T - t0))
    ) + sig * sig * 0.25 / k * (
        1 - np.exp(-2 * k * (T - t0))
    ) * seasonality_function(
        T, c1, c2, c3, c4
    )


def xtdt(x0, a, sig, k, dt, w):
    return B(k, dt) * x0 + g(a, k, dt) + np.sqrt(R(sig, k, dt) * dt) * w


def generate_path(x0, a, l, sig, k, dt, Tn, c1, c2, c3, c4, eta, n_steps=100):
    n_fut = len(Tn)
    w = np.random.normal(size=n_steps)
    q = np.random.normal(size=(n_fut, n_steps))
    spot = np.zeros(shape=n_steps + 1)
    futures = np.zeros(shape=(n_fut, n_steps))
    spot[0] = x0
    T = np.array(Tn)
    for i, W in enumerate(w):
        spot[i + 1] = xtdt(spot[i], a, sig, k, dt, w[i])
        futures[:, i] = (
            An(T, k) * spot[i]
            + dn(a, l, sig, k, T, c1, c2, c3, c4)
            + q[:, i] * eta
        )
        for i, t in enumerate(list(T)):
            if T[i] - dt >= 0:
                T[i] -= dt
            else:
                T[i] = Tn[i]
    return spot, futures


def one_factor_model():
    pass


if __name__ == "__main__":
    x0, alpha, lmbd, sigma, kappa, dt = 30, 0.0658, 0, 0.43, 5, 1 / 252.0
    generate_path(x0, alpha, lmbd, sigma, kappa, dt)
