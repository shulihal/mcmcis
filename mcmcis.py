import numpy as np
from tqdm.auto import tqdm

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
        elif is_func=='poly':
            return beta ** (x-xzero)
        else:
            return 1.0

def mcmc_step():
    return 1


def mcmcis(lambdaStar, L, X1, X2, is_func, t0,
           beta=0, adaptive=False, pi=0.5, window = 1 ,frac = 1,
           K=10**5, J=10**2, Ti=10**4):
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
        if adaptive and j>=window-1:
            # \beta=log(-\pi K+\sum_{i=1}^{K}I\{x_{i+1}\ge x^{*}|x_{i}\}+I\{x_{i+1}<x^{*}|x_{i}\ge x^{*}\})/\sum_{i+1\in\{x_{i+1}<x^{*}|x_{i}\ge x^{*}\}}x_{i+1}-x^{*}

            # beta=np.log(-pi*K+(x_i==-1).sum())/(x_i[x_i>0].sum())

            pi_hat = (theta11[j+1-window:j+1,:]!=0).sum()/ (K*window)
            # beta += gamma(j, t0)*np.log((1-pi_hat)/(1-pi))
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
    pi_hat = (theta11[-1,:]!=0).sum()/K
    return res, j+1, beta, accept_rate, up_rate, pi_hat
