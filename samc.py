import numpy as np
from tqdm.auto import tqdm

import mh

def gamma(t, t0=1e4):
    return t0 / max(t0, t)

def lambdak(k, m, lambdaStar, lambda0=0):
    return lambda0 + k * (lambdaStar - lambda0) / (m - 1)

def bounds(m, lambdaStar, lambda0=0):
    lower = np.array([lambdak(x, m, lambdaStar, lambda0) for x in range(m)])
    upper = np.roll(lower, -1)
    upper[-1] = np.inf
    return lower, upper


def mcmc_step(X1, X2, L, sum_diff_x, lower_bound, upper_bound, theta, i, pi, w_update, kX, t0):
    Y1, Y2, d = mh.propose(X1, X2, L)
    sum_diff_y = sum_diff_x + 2 * d
    lambdaY = abs(sum_diff_y)
    kY = np.where((lambdaY >= lower_bound) & (lambdaY < upper_bound))[0][0]
    r = np.exp(theta[kX] - theta[kY])
    acc = 0
    if np.random.rand() < r:
        X1, X2 = Y1.copy(), Y2.copy()
        kX, sum_diff_x = kY, sum_diff_y
        acc = 1

    theta -= gamma(i, t0) * pi
    if w_update == '0':
        theta[kX] += gamma(i, t0) * pi
    else:
        theta[kX] += gamma(i, t0)

    return X1, X2, kX, sum_diff_x, theta, acc

def samc(lambdaStar, L, X1, X2, m=101, w_update = 'base', T=2e5, K=5e6, t0=1e4):
    pi = 1 / m
    lower_bound, upper_bound = bounds(m, lambdaStar)

    sum_diff_x = mh.sum_diff(X1, X2)
    lambdaX = abs(sum_diff_x)
    kX = np.where((lambdaX >= lower_bound) & (lambdaX < upper_bound))[0][0]
    theta = np.zeros(m)

    for t in range(T):
        X1, X2, kX, sum_diff_x, theta, acc = mcmc_step(X1, X2, L, sum_diff_x, lower_bound, upper_bound, theta, t, pi, w_update, kX, t0)

    theta.fill(0)
    upper = 0
    accept = 0

    for k in tqdm(range(int(K))):
        X1, X2, kX, sum_diff_x, theta, acc = mcmc_step(X1, X2, L, sum_diff_x, lower_bound, upper_bound, theta, k, pi, w_update, kX, t0)
        accept += acc
        upper += kX == m-1
    
    pval = 1 / (np.sum(np.exp(theta - theta[-1])))
    accept_rate = accept / K
    uprate = upper / K
    return pval, K, accept_rate, uprate
