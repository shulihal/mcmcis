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
    for _ in range(M):
        indices = np.random.choice(200, 100, replace=False)
        y1 = X12[indices].sum()
        y2 = sum_total - y1
        lam_sum += abs(y1 - y2)
    rho = M / lam_sum
    return rho


def h_func(beta, lam_star, rho, pi_tilda, is_func):
    def integrand_f(x):
        return np.exp(- (rho**2 * x**2) / np.pi)
    
    def integrand_joint(x):
        if is_func=='exp':
            return np.exp(beta * x- (rho**2 * x**2) / np.pi)
        elif is_func=='poly':
            return (beta**(x - lam_star)) * np.exp(- (rho**2 * x**2) / np.pi)
    
    integral_f, _ = quad(integrand_f, lam_star, np.inf)
    integral_fg, _ = quad(integrand_joint, 0, lam_star)

    if is_func =='exp':
        integral_fg =  (2 * rho * np.exp(- beta * lam_star) * integral_fg) / np.pi 
        integral_f = (2 * rho * integral_f) / np.pi

    return integral_f / (integral_f + integral_fg) - pi_tilda


def dh_calc(rho, lam_star, beta, is_func):
    def integrand_f(x):
        return np.exp(- (rho**2 * x**2) / np.pi)
    
    integral_f, _ = quad(integrand_f, lam_star, np.inf)
    
    def integrand_joint(x):
        return np.exp(beta * x- (rho**2 * x**2) / np.pi)

    def integrand_fgx(x):
        return x * np.exp(beta * x- (rho**2 * x**2) / np.pi)

    if is_func=='exp':
        integral_joint, _ = quad(integrand_joint, 0, lam_star)
        integral_fgx, _ = quad(integrand_fgx, 0, lam_star)

        numerator = (lam_star * integral_joint - integral_fgx) * np.exp(beta * lam_star) * integral_f
        denominator = (np.exp(beta * lam_star) * integral_f + integral_joint)**2
    
    elif is_func=='poly':
        def integrand2(x):
            return beta**(x - lam_star - 1) * (x - lam_star) * np.exp(- (rho**2 * x**2) / np.pi)

        def integrand3(x):
            return beta**(x - lam_star) * np.exp(- (rho**2 * x**2) / np.pi)

        integral2, _ = quad(integrand2, 0, lam_star)
        integral3, _ = quad(integrand3, 0, lam_star)

        numerator = - (integral_f * integral2)
        denominator = (integral_f + integral3)**2
    
    return numerator / denominator


def mcmcis(lambdaStar, L, X1, X2, is_func, t0,
           beta=0, adaptive=False, pi=0.01, frac = 1,
           K=10**5, J=10**2, Ti=10**4):
    rho = random_sampling_lambda(X1, X2)
    beta_solution = fsolve(h_func, beta, args=(lambdaStar, rho, pi, is_func))
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
            dh = dh_calc(rho, lambdaStar, beta, is_func)
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
    pi_hat = (theta11[-1,:]!=0).sum()/K

    return res, j+1, beta, accept_rate, up_rate, pi_hat
