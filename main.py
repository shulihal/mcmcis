import multiprocessing
import argparse
import numpy as np
import pandas as pd
import os
from time import time

import mh
import store_results

from mcmcis import mcmcis
from samc import samc


def execute(arg):
    X1 = np.loadtxt('data/data.txt', dtype='int')[2 * arg.exm_id]
    X2 = np.loadtxt('data/data.txt', dtype='int')[2 * arg.exm_id + 1]

    true_val = mh.perm_exact_pval_diff(X1, X2)
    lambdaStar = abs(mh.sum_diff(X1, X2))
    L = 5

    results = pd.DataFrame(columns=['algo', 'example_id', 'true_val', 'result',
                                    'beta', 'gamma', 'adaptive', 'pi', 'window', 'fraction', 'IS_func',
                                    'accept_rate', 'up_rate', 'pi_hat', 'runtime', 'iterations',
                                    'T', 'K', 'J', 'notes'])

    for _ in range(arg.n_runs // arg.num_processes):
        start_time = time()
        if arg.algo == 'samc':
            if arg.w_func != 'exp':
                w_update = arg.w_func
            else:
                w_update = 'base'
            m = int(arg.beta)
            res, j, accrate, up_rate = samc(lambdaStar, L, X1, X2,
                                                    m, w_update,
                                                    arg.T, arg.K, arg.gamma)
            is_func, iterations, beta, pi_hat = w_update,  arg.K+ arg.T, m, up_rate
        else:
            res, j, beta, accrate, up_rate, pi_hat = mcmcis(lambdaStar, L, X1, X2, arg.w_func, arg.gamma,
                                                    arg.beta, arg.adaptive, arg.pi, arg.window, arg.frac,
                                                    arg.K, arg.J, arg.T)
            is_func, iterations = arg.w_func, j*(arg.K+arg.T)
        end_time = time()
        runtime = end_time - start_time

        store_results.insert_result(arg.algo, arg.exm_id, true_val, res,
                                    beta, arg.gamma, arg.adaptive, arg.pi, arg.window, arg.frac, is_func, 
                                    accrate, up_rate, pi_hat, runtime, iterations,
                                    arg.T, arg.K, arg.J, notes=arg.notes)

        results.loc[results.shape[0]] = [arg.algo, arg.exm_id, true_val, res,
                                         beta, arg.gamma, arg.adaptive, arg.pi, arg.window, arg.frac, is_func,
                                         accrate, up_rate, pi_hat, runtime, iterations,
                                         arg.T, arg.K, arg.J, arg.notes]

    return results


def main():
    parser = argparse.ArgumentParser(description='Execute the algorithms in parallel.')
    parser.add_argument('--algo', choices=['samc', 'mcmcis'], required=True, help='Algorithm to run.')
    parser.add_argument('--exm_id', type=int, required=True, help='Example ID.')
    parser.add_argument('--T', type=int, required=True, help='T parameter.')
    parser.add_argument('--K', type=int, required=True, help='K parameter.')
    parser.add_argument('--J', type=int, required=False, default=1, help='J parameter.')
    parser.add_argument('--beta', type=float, required=True, help='Beta value.')
    parser.add_argument('--gamma', type=float, required=False, default=1, help='t0 for gamma value.')
    parser.add_argument('--w_func', type=str, required=False, default='exp', help='IS func for mcmcis, weight update for SAMC')
    parser.add_argument('--adaptive', type=bool, required=False, default=False, help='Adaptive parameter.')
    parser.add_argument('--pi', type=float, required=False, default=None, help='tail')
    parser.add_argument('--window', type=int, required=False, default=1, help='window for beta update')
    parser.add_argument('--frac', type=float, required=False,default=1, help='sampling fraction')
    parser.add_argument('--n_runs', type=int, required=True, help='Number of runs.')
    parser.add_argument('--num_processes', type=int,default=multiprocessing.cpu_count(), help='Number of processes to run in parallel.')
    parser.add_argument('--notes', nargs='?', required=False, default=None, help='Additional notes.')
    arg = parser.parse_args()

    if arg.n_runs < arg.num_processes:
        arg.num_processes = arg.n_runs
    print(arg)

    results_list = []
    with multiprocessing.Pool(processes=arg.num_processes) as pool:
        results_list = pool.map(execute, [arg] * arg.num_processes)

    if os.path.exists('data/results.csv'):
        main_results = pd.read_csv('data/results.csv', on_bad_lines='skip')
    else:
        main_results = pd.DataFrame()

    for res in results_list:
        main_results = pd.concat([main_results, res])

    main_results.to_csv('data/results.csv', index=False)


if __name__ == '__main__':
    main()
