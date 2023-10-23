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


def execute(arguments):
    X1 = np.loadtxt('data/data.txt', dtype='int')[2 * arguments.exm_id]
    X2 = np.loadtxt('data/data.txt', dtype='int')[2 * arguments.exm_id + 1]

    true_val = mh.perm_exact_pval_diff(X1, X2)
    lambdaStar = abs(mh.sum_diff(X1, X2))
    L = 5

    results = pd.DataFrame(columns=['algo', 'example_id', 'true_val', 'result',
                                    'beta', 'adaptive', 'IS_func',
                                    'accept_rate', 'up_rate', 'runtime', 'iterations',
                                    'T', 'K', 'J', 'notes'])

    for _ in range(arguments.n_runs // arguments.num_processes):
        start_time = time()
        if arguments.algo == 'samc':
            m = int(arguments.beta)
            res, j, accrate, up_rate = samc(lambdaStar, L, X1, X2,
                                                    m, arguments.w_func,
                                                    arguments.T, arguments.K)
            is_func, iterations, beta = arguments.w_func,  arguments.K+ arguments.T, m
        else:
            res, j, beta, accrate, up_rate = mcmcis(lambdaStar, L, X1, X2,
                                                    arguments.beta, arguments.adaptive,
                                                    arguments.K, arguments.J, arguments.T)
            is_func, iterations = 'exp', j*(arguments.K+arguments.T)
        end_time = time()
        runtime = end_time - start_time

        store_results.insert_result(arguments.algo, arguments.exm_id, true_val, res,
                                    beta, arguments.adaptive, is_func,
                                    accrate, up_rate, runtime, iterations,
                                    arguments.T, arguments.K, arguments.J, notes=arguments.notes)

        results.loc[results.shape[0]] = [arguments.algo, arguments.exm_id, true_val, res,
                                         arguments.beta, arguments.adaptive, is_func,
                                         accrate, up_rate, runtime, iterations,
                                         arguments.T, arguments.K, arguments.J, arguments.notes]

    return results


def main():
    parser = argparse.ArgumentParser(description='Execute the algorithms in parallel.')
    parser.add_argument('--algo', choices=['samc', 'mcmcis'], required=True, help='Algorithm to run.')
    parser.add_argument('--exm_id', type=int, required=True, help='Example ID.')
    parser.add_argument('--T', type=int, required=True, help='T parameter.')
    parser.add_argument('--K', type=int, required=True, help='K parameter.')
    parser.add_argument('--J', type=int, required=False, default=1, help='J parameter.')
    parser.add_argument('--beta', type=float, required=True, help='Beta value.')
    parser.add_argument('--w_func', type=str, required=False, help='IS func for mcmcis, weight update for SAMC')
    parser.add_argument('--adaptive', choices=[True, False], required=False, default=False, help='Adaptive parameter.')
    parser.add_argument('--n_runs', type=int, required=True, help='Number of runs.')
    parser.add_argument('--num_processes', type=int,default=multiprocessing.cpu_count(), help='Number of processes to run in parallel.')
    parser.add_argument('--notes', nargs='?', required=False, default=None, help='Additional notes.')
    arguments = parser.parse_args()

    if arguments.n_runs < arguments.num_processes:
        arguments.num_processes = arguments.n_runs
    print(arguments)

    results_list = []
    with multiprocessing.Pool(processes=arguments.num_processes) as pool:
        results_list = pool.map(execute, [arguments] * arguments.num_processes)

    if os.path.exists('data/results.csv'):
        main_results = pd.read_csv('data/results.csv', on_bad_lines='skip')
    else:
        main_results = pd.DataFrame()

    for res in results_list:
        main_results = pd.concat([main_results, res])

    main_results.to_csv('data/results.csv', index=False)


if __name__ == '__main__':
    main()
