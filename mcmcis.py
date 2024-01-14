import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import fsolve
from scipy.stats import norm

import mh

def gamma(t, t0=1):
    return t0 / max(t0, t)

def g_func(xzero, beta, x, is_func): # trial function
    if x >= xzero:
        return 1.0
    else:
        if is_func=='exp':
            return np.exp(beta * (x - xzero))
        elif is_func=='power':
            return (x / xzero) ** beta
        elif is_func=='sigmoid':
            return 2 / (1 + np.exp(-beta * (x - xzero)))
        else:
            return 1.0

def mcmc_step():
    return 1


def random_sampling_lambda(x1, x2, M=10**5):
    X12 = np.concatenate([x1, x2])
    sum_total = X12.sum()
    xs = np.zeros(M)
    for m in range(M):
        indices = np.random.choice(200, 100, replace=False)
        y1 = X12[indices].sum()
        xs[m] = 2*y1 - sum_total

    return xs.std(ddof=1)


def h_func(beta, x_star, sigma, pi_tilda):
    tail = 1 - norm.cdf(x_star, 0, sigma)

    phi1 = norm.cdf(x_star, beta * sigma**2, sigma)
    phi2 = norm.cdf(0, beta * sigma, 1)
    exponential = np.exp(beta * (0.5*beta*sigma**2 - x_star))
    body =  (phi1-phi2) * exponential

    return tail / (tail + body) - pi_tilda


def dh_func(beta, x_star, sigma):
    tail = 1 - norm.cdf(x_star, 0, sigma)

    phi1 = norm.cdf(x_star, beta * sigma**2, sigma)
    phi2 = norm.cdf(0, beta * sigma, 1)
    exponential = np.exp(beta * (0.5*beta*sigma**2 - x_star))
    B =  (phi1-phi2) * exponential

    exponential2 = np.exp(-0.5 * sigma**2 * beta**2) - np.exp(-0.5*((x_star - sigma**2 * beta)**2)/sigma**2)
    A = sigma**2 * beta * (phi1 - phi2) + exponential2 * sigma / np.sqrt(2* np.pi)
    dB = exponential * A - x_star * B

    return - (dB * tail) / (tail + B)**2

def mcmcis(lambdaStar, L, X1, X2, is_func, t0,
           beta=0, adaptive=False, pi=0.01, frac = 1,
           K=10**5, J=10**2, Ti=10**4):
    sigma = random_sampling_lambda(X1, X2)
    beta_solution = fsolve(h_func, beta, args=(lambdaStar, sigma, pi))
    beta =  beta_solution[0]

    accept = 0
    X1new = X1.copy()
    X2new = X2.copy()
    sum_diff_x = mh.sum_diff(X1new, X2new)
    lambdaX = abs(sum_diff_x)
    gX = g_func(xzero= lambdaStar, beta =beta, x=lambdaX, is_func=is_func)
    theta10 = np.zeros(shape=[J,K])
    theta11 = np.zeros(shape=[J,K])

    for j in tqdm(range(J)):
        for _ in range(Ti): #burnin
            Y1, Y2, d = mh.propose(X1new, X2new, L)
            sum_diff_y = sum_diff_x + 2*d
            lambdaY = abs(sum_diff_y)
            gY = g_func(xzero= lambdaStar, beta =beta, x=lambdaY, is_func=is_func)

            p = gY/gX
            q = np.random.rand()
            if q<p:   #accept
                X1new = Y1.copy()
                X2new = Y2.copy()
                sum_diff_x = sum_diff_y.copy()
                gX = gY
                accept += 1
        lambdaX = abs(sum_diff_x)
        
        for k in tqdm(range(K)):
            Y1, Y2, d = mh.propose(X1new, X2new, L)
            sum_diff_y = sum_diff_x + 2*d
            lambdaY = abs(sum_diff_y)
            gY = g_func(xzero= lambdaStar, beta =beta, x=lambdaY, is_func=is_func)
            
            p = gY/gX
            q = np.random.rand()
            if q<p:   #accept
                X1new = Y1.copy()
                X2new = Y2.copy()
                gX = gY
                lambdaX = lambdaY.copy()
                sum_diff_x = sum_diff_y.copy()
                accept += 1

            #weight update
            theta10[j,k] = 1/gX
            if lambdaX >= lambdaStar:
                theta11[j,k] = 1/gX

        #parameter beta update
        if adaptive and J>1:
            pi_hat = (theta11[j,:]>0).sum()/ K
            dh = dh_func(beta, lambdaStar, sigma)
            beta -= (pi_hat - pi) / dh

            if beta<0:
                beta=0
    
    theta0 = 0
    theta1 = 0
    for row in range(J):
        sampled_indices = np.random.choice(K, size=int(K*frac), replace=False)
        theta0  += theta10[row,sampled_indices].sum()
        theta1  += theta11[row,sampled_indices].sum()

    res = (theta1/theta0)
    iter = (j+1)*(K+Ti)
    accept_rate = accept / iter
    up_rate = (theta11>0).sum()/ (K*J)
    pi_hat = (theta11[-1,:]>0).sum()/K

    return res, j+1, beta, accept_rate, up_rate, pi_hat
