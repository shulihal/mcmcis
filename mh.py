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

    return sum(rtab[N-1,:,nl])/math.comb(N,nl)


## test statistic (lambda).
def lambda_sum_diff_abs(x, y): 
    return abs(sum(y)-sum(x))


def propose(x, y, l): #q(x,*)
    nx = len(x)
    ny = len(y)

    for _ in range(l):
        repx = np.random.randint(0,nx,1)
        valx = x[repx]

        repy = np.random.randint(0,ny,1)
        valy = y[repy]

        x[repx] = valy
        y[repy] = valx

    return x,y
