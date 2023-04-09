import numpy as np
from tqdm.auto import tqdm

import mh


def g_func(x, beta, xzero): # trial function
    if x >= xzero:
        return 0.5
    else:
        return 0.5*np.exp(beta*(x-xzero))
    # return 1 / (1 + np.exp( -beta*(x - xzero)))


def mcmcis(lambdaStar, l, X1, X2, alpha, K=10**5, J=10**2, Ti=10**4):
    a=15
    beta=0.5
    gam = 0.1
    X1new = X1
    X2new = X2
    lambdaX = mh.lambda_sum_diff_abs(X1new, X2new)
    gX = g_func(x=lambdaX, beta =beta, xzero=lambdaStar)
    theta10 = np.zeros(shape=[J,K])
    theta11 = np.zeros(shape=[J,K])
    for j in tqdm(range(J)):
        for t in range(Ti): #burnin
            Y1, Y2 = mh.propose(X1new, X2new, l)
            lambdaY = mh.lambda_sum_diff_abs(Y1, Y2)
            gY = g_func(xzero= lambdaStar, beta =beta, x=lambdaY)

            p = gY/gX
            q = np.random.rand()
            if q<p:   #accept
                X1new = Y1.copy()
                X2new = Y2.copy()
                gX = gY

        for k in range(K):
            Y1, Y2 = mh.propose(X1new, X2new, l)
            lambdaY = mh.lambda_sum_diff_abs(Y1, Y2)
            gY = g_func(xzero= lambdaStar, beta =beta, x=lambdaY)

            p = gY/gX
            q = np.random.rand()
            if q<p:   #accept
                X1new = Y1.copy()
                X2new = Y2.copy()
                gX = gY
                lambdaX = lambdaY

            #weight update
            theta10[j,k] = 1/gX
            if lambdaX >= lambdaStar:
                theta11[j,k] = 1/gX

        if j>=a and j%10==0:
            pvals = (theta11[:j+1,:].sum(axis=1).cumsum()/theta10[:j+1,:].sum(axis=1).cumsum())
            if pvals[j-a:].std() < pvals[-1]*alpha:
                break

        #parameter beta update
        pi = (theta11[j,:]!=0).sum()/ K
        beta += gam*(0.5-pi)

    return (theta11.sum()/theta10.sum()), (j+1)*K





