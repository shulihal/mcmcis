import numpy as np
import math


# checking the exact pval
def perm_exact_pval_diff(X,Y):
    XY = np.concatenate([X,Y])
    N = len(XY)
    nl = len(X)
    sl = sum(X)

    rtab = np.zeros(shape=[N,sl+1,nl+1])
    rtab[0,XY[0],1]=1
    rtab[0,0,0]=1
    for i in range(1,N):
        if sl-XY[i] > 0:
            rtab[i, XY[i]:sl+1, 1:min(nl,i+1)+1] = rtab[i-1,0:sl-XY[i]+1,0:min(nl-1,i)+1]
        rtab[i,:,:] = rtab[i,:,:] + rtab[i-1, :, :]

    return 2*sum(rtab[N-1,:,nl])/math.comb(N,nl)


## test statistic (lambda).
def sum_diff(x, y):
    return y.sum() - x.sum()


def propose(X, Y, L):
    x = X.copy()
    y = Y.copy()

    repx = np.random.choice(len(x), L, replace=False)
    repy = np.random.choice(len(y), L, replace=False)
    valx = x[repx]
    valy = y[repy]

    x[repx], y[repy] = valy, valx
    d = valx.sum() - valy.sum()

    return x, y, d

