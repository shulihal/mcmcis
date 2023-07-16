import multiprocessing
from sys import argv
from time import time
import numpy as np

from mcmcis import mcmcis
from samc import samc
import mh
import store_results

# mcmcis: algo exampleid T K J alpha iter npros notes

# samc: algo exampleid T m J alpha iter npros notes

def execute():
    exm_id = int(argv[2])
    X1 = np.loadtxt('data.txt', dtype='int')[2*exm_id]
    X2 = np.loadtxt('data.txt', dtype='int')[2*exm_id+1]

    algo = argv[1]
    true_val = mh.perm_exact_pval_diff (X1,X2)
    lambdaStar = mh.lambda_sum_diff_abs(X1,X2)
    l=5
    
    T = int(argv[3])
    K = int(argv[4])
    J = int(argv[5])
    alpha=float(argv[6])

    if len(argv)>9:
        note = argv[9]
    else:
        note = None

    for _ in range(int(argv[7])):
        start_time = time()
        
        if algo =='samc':
            res, iter = samc(lambdaStar, l, X1, X2, alpha, m=K, T=T, J=J)
            
            end_time = time()
            runtime = end_time -   start_time

            store_results.insert_result(algo, exm_id, runtime, true_val, res, iter, alpha, note=note, T=T, m=K)

        elif algo == 'mcmcis':
            res, j, iter = mcmcis(lambdaStar, l, X1, X2, alpha, K, J, T)

            end_time = time()
            runtime = end_time -   start_time

            store_results.insert_result(algo, exm_id, runtime, true_val, res, iter, alpha, note=note, IS_func='half_exp', T=T, K=K, J=j)

        else:
            print(f'No such algorithm {algo}')
            break

if __name__ == '__main__':
    processes = []
    num_processes = int(argv[8]) # number of processes to run in parallel
    for i in range(num_processes):
        p = multiprocessing.Process(target=execute)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

