import numpy as np
from tqdm.auto import tqdm

import mh

def gamma(t, t0=1e5):
    return t0 / max(t0, t)

def lambdak(k, m, lambdaStar, lambda0=0):
    return lambda0 + k * (lambdaStar - lambda0) / (m - 1)

def bounds(m, lambdaStar, lambda0=0):
    lower = np.array([lambdak(x, m, lambdaStar, lambda0) for x in range(m)])
    upper = np.roll(lower, -1)
    upper[-1] = np.inf
    return lower, upper


def samc(lambdaStar, L, X1, X2, m=101, T=2e5, K=5e6, alpha=0.2):
    pi = 1 / m
    lower_bound, upper_bound = bounds(m, lambdaStar)

    sum_diff_x = mh.sum_diff(X1, X2)
    lambdaX = abs(sum_diff_x)
    kX = np.where((lambdaX >= lower_bound) & (lambdaX < upper_bound))[0][0]
    theta = np.zeros(m)

    for t in range(T):
        Y1, Y2, d = mh.propose(X1, X2, L)
        sum_diff_y = sum_diff_x + 2 * d
        lambdaY = abs(sum_diff_y)
        kY = np.where((lambdaY >= lower_bound) & (lambdaY < upper_bound))[0][0]
        r = np.exp(theta[kX] - theta[kY])
        if np.random.rand() < r:
            X1, X2 = Y1.copy(), Y2.copy()
            kX, sum_diff_x = kY, sum_diff_y

        theta -= gamma(t) * pi
        theta[kX] += gamma(t) * pi

    theta.fill(0)
    upper = 0
    accept = 0

    for j in tqdm(range(int(K))):
        Y1, Y2, d = mh.propose(X1, X2, L)
        sum_diff_y = sum_diff_x + 2 * d
        lambdaY = abs(sum_diff_y)
        kY = np.where((lambdaY >= lower_bound) & (lambdaY < upper_bound))[0][0]
        r = np.exp(theta[kX] - theta[kY])
        if np.random.rand() < r:
            X1, X2 = Y1.copy(), Y2.copy()
            kX, sum_diff_x = kY, sum_diff_y
            accept += 1

        upper += kX == m-1
        theta -= gamma(j) * pi
        theta[kX] += gamma(j) * pi
    
    pval = 1 / (np.sum(np.exp(theta - theta[-1])))
    accept_rate = accept / K
    uprate = upper / K
    return pval, K, m, accept_rate, uprate
