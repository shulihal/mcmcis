import numpy as np
from tqdm.auto import tqdm

import mh


def g_func(xzero, beta, x): # trial function
    return 1.0 if x >= xzero else np.exp(beta * (x - xzero))
    # return 2 / (1 + np.exp( -beta*(x - xzero)))
    # return (x/xzero)**(beta)


def mcmcis(lambdaStar, L, X1, X2, 
           beta=0, adaptive=False,
           K=10**5, J=10**2, Ti=10**4):
    accept = 0
    gam = 0.1
    X1new = X1.copy()
    X2new = X2.copy()
    sum_diff_x = mh.sum_diff(X1new, X2new)
    lambdaX = abs(sum_diff_x)
    gX = g_func(xzero= lambdaStar, beta =beta, x=lambdaX)
    theta10 = np.zeros(shape=[J,K])
    theta11 = np.zeros(shape=[J,K])
    for j in tqdm(range(J)):
        for _ in range(Ti): #burnin
            Y1, Y2, d = mh.propose(X1new, X2new, L)
            sum_diff_y = sum_diff_x + 2*d
            lambdaY = abs(sum_diff_y)
            gY = g_func(xzero= lambdaStar, beta =beta, x=lambdaY)

            p = gY/gX
            q = np.random.rand()
            if q<p:   #accept
                X1new = Y1.copy()
                X2new = Y2.copy()
                sum_diff_x = sum_diff_y.copy()
                gX = gY
                accept += 1

        for k in range(K):
            Y1, Y2, d = mh.propose(X1new, X2new, L)
            sum_diff_y = sum_diff_x + 2*d
            lambdaY = abs(sum_diff_y)
            gY = g_func(xzero= lambdaStar, beta =beta, x=lambdaY)

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
        if adaptive:
            pi = (theta11[j,:]!=0).sum()/ K
            beta += gam*(0.5-pi)
    res = (theta11.sum()/theta10.sum())
    iter = (j+1)*(K+Ti)
    accept_rate = accept / iter
    up_rate = (theta11!=0).sum()/ K*J
    return res, j+1, beta, accept_rate, up_rate
