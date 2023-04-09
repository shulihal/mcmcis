import multiprocessing
from sys import argv
from time import time
import numpy as np

from mcmcis import mcmcis
from samc import samc
import mh
import store_results

# argv: [main.py, algo, alpha, exp_id, iter, n_pross, notes]

def execute():
    exm_id = int(argv[3])
    X1 = np.loadtxt('data.txt', dtype='int')[2*exm_id]
    X2 = np.loadtxt('data.txt', dtype='int')[2*exm_id+1]

    algo = argv[1]
    true_val = mh.perm_exact_pval_diff (X1,X2)
    lambdaStar = mh.lambda_sum_diff_abs(X1,X2)
    l=5
    alpha=float(argv[2])

    if len(argv)>6:
        note = argv[6]
    else:
        note = None

    for _ in range(int(argv[4])):
        start_time = time()
        
        if algo =='samc':
            res, iter = samc(lambdaStar, l, X1, X2, alpha)
        elif algo == 'mcmcis':
            res, iter = mcmcis(lambdaStar, l, X1, X2, alpha)
        else:
            print(f'No such algorithm {algo}')
            break

        end_time = time()
        runtime = end_time - start_time
        
        store_results.insert_result(algo, exm_id, alpha, runtime, true_val, res, iter, note)

if __name__ == '__main__':
    processes = []
    num_processes = int(argv[5]) # number of processes to run in parallel
    for i in range(num_processes):
        p = multiprocessing.Process(target=execute)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if alpha<0:
    
    argv[7]