import numpy as np
from tqdm.auto import tqdm
from scipy.integrate import quad
from scipy.optimize import fsolve

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


def random_sampling_lambda(x1, x2, M=10**5):
    X12 = np.concatenate([x1, x2])
    sum_total = X12.sum()
    lam_sum = 0
    for k in range(M):
        indices = np.random.choice(200, 100, replace=False)
        y1 = X12[indices].sum()
        y2 = sum_total - y1
        lam_sum += abs(y1 - y2)
    rho = M / lam_sum
    return rho


def integrand_norm(x, sigma):
    return np.exp(- (sigma**2 * x**2) / np.pi)

def compute_integral_norm(x_star, sigma):
    integral, _ = quad(integrand_norm, x_star, np.inf, args=(sigma,))
    return (2 * sigma * integral) / np.pi

def integrand_joint(x, beta, sigma):
    return np.exp(beta * x- (sigma**2 * x**2) / np.pi)

def equation(beta, x_star, sigma, pi_tilda):
    integral, _ = quad(integrand_joint, 0, x_star, args=(beta, sigma))
    with_gx =  (2 * sigma * np.exp(- beta * x_star) * integral) / np.pi 
    fx_int = compute_integral_norm(x_star, sigma)
    return fx_int / (fx_int + with_gx) - pi_tilda


def dh(sigma, x_star, beta):
    def integrand1(x):
        return np.exp(beta * x- (sigma**2 * x**2) / np.pi)

    def integrand2(x):
        return x * np.exp(beta * x- (sigma**2 * x**2) / np.pi)

    def integrand3(x):
        return np.exp(- (sigma**2 * x**2) / np.pi)

    integral1, _ = quad(integrand1, 0, x_star)
    integral2, _ = quad(integrand2, 0, x_star)
    integral3, _ = quad(integrand3, x_star, np.inf)

    numerator = (x_star * integral1 - integral2) * np.exp(beta * x_star) * integral3
    denominator = (np.exp(beta * x_star) * integral3 + integral1)**2

    return numerator / denominator


def mcmcis(lambdaStar, L, X1, X2, is_func, t0,
           beta=0, adaptive=False, pi=0.01, frac = 1,
           K=10**5, J=10**2, Ti=10**4):
    rho = random_sampling_lambda(X1, X2)
    beta_solution = fsolve(equation, beta, args=(lambdaStar, rho, pi))
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
        if adaptive:
            pi_hat = (theta11[j,:]!=0).sum()/ K
            d_h = dh(rho, lambdaStar, beta)
            beta -= (pi_hat - pi) / d_h

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
