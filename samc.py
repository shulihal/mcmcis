import numpy as np
import pandas as pd
import math
from tqdm.auto import tqdm

import mh


def gamma(t, t0=10000):
    return t0/max(t0,t)


def lambdak(k, m, lambdaStar, lambda0):
    return lambda0 + k*(lambdaStar-lambda0)/(m-1)

def bounds(m, lambdaStar, lambda0=0):
    parts = pd.Series(range(m))
    Ebound = pd.DataFrame(columns=['lower', 'upper'])
    Ebound['lower'] = parts.apply(lambda x: lambdak(x,m,lambdaStar, lambda0))
    Ebound['upper'] = parts.apply(lambda x: lambdak(x+1,m,lambdaStar, lambda0))
    Ebound.loc[m-1,'upper'] = math.inf
    return Ebound


def pvalue_calc(Theta, e, m, pi):
    m0 = sum(e==0)
    pi0 = sum(pi[e==0]) / (m-m0)

    return np.exp(Theta[m-1])*(pi[m-1]+pi0) / sum(np.exp(Theta)*(pi + pi0))

def rsfe(E, j, m):
    m0 = sum(E==0)
    epsilon = (E/j - 1/(m-m0)) / (1/(m-m0))
    epsilon = abs(epsilon)
    epsi = epsilon[epsilon!=1]
    return max(epsi) 


def samc(lambdaStar, l, X1, X2, alpha=0.2, m=101):
    theta = np.zeros(m)
    E = np.zeros(m)
    pi = np.array([1/m]*m)
    l=5
    X1new = X1
    X2new = X2
    lambdaX = mh.lambda_sum_diff_abs(X1new, X2new)
    Ebound = bounds(m, lambdaStar)
    kX = Ebound.query('@lambdaX >= lower and @lambdaX < upper').index
    J=5*10**6
    
    for j in tqdm(range(J)):
        Y1, Y2 = mh.propose(X1new, X2new, l)
        lambdaY = mh.lambda_sum_diff_abs(Y1, Y2)
        kY = Ebound.query('@lambdaY >= lower and @lambdaY < upper').index

        #ratio
        r = np.exp(theta[kX]-theta[kY])
        q = np.random.rand()
        if q<r:   #accept
            X1new = Y1.copy()
            X2new = Y2.copy()
            kX = kY.copy()

        E[kX] += 1
        #weight update
        theta -= gamma(j)*pi
        theta[kX] += gamma(j)*pi[kX]
        if j % 100 == 0 and j>=200000:
            if rsfe(E, j, m) < alpha: #stopping rule
                break
    return pvalue_calc(theta, E, m, pi), j



