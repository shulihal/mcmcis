# MCMCIS Algorithm
### p-value estimation for permutation tests

An efficient algorithm for estimating tail in unknown distribution, using Markov Chain Monte Carlo and Importance Sampling techniques.
Compared to the SAMC algorithm and different IS functions.

For running via teminal run: 'python3 main.py ALGORITHM EXAMPLE_ID ALPHA ITERATIONS NOTES'

## General Idea

Our objective is to determine the size of the tail of a given distribution, the probability of sampling from that region. Ideally, we could achieve this by randomly sampling from the space and then weighting the samples from the tail over the rest of the space (i.e., the resampling method). However, for tails that are extremely small, such as those in the size of $10^{-7}$, we would need an enormous number of samples to attain statistical confidence - on the order of $10^{9}$ iterations.

In our approach, we aim to increase the probability of sampling from the tail region, despite its minuscule size. To achieve this, we employ Importance Sampling (IS) with a trial function in which the probability of sampling from the tail is greater than that of the rest of the function. In this way, we can obtain sufficient samples with a lower number of iterations. For each sample, we assign an IS weight using the IS function, such that each sample has a weight that represents the level of difficulty in sampling it.

We sample using MCMC, specifically the Metropolis-Hastings algorithm. During each iteration, a new step is proposed (e.g., a permutation with a stochastic change of the previous step), and the weighted value of the trial function is used to decide whether to accept or reject the proposed step.

In the final step, we calculate the p-value by summing the IS weights obtained from the tail region and dividing by the total of all sampled weights.

## The Algorithm

Let $x$ be a sample in the sample space $X$, where $g_{\beta}(x)$ is the IS trial function with scaling parameter $\beta$, and $\psi(x)$ is the sample distribution. At each iteration of the Metropolis-Hastings MCMC, a new step is proposed, and we use the IS weight, $\frac{\psi(x)}{g(x)}$, for the probabilities ratio to decide whether to accept the new step. The algorithm consists of three parts: (1) burn-in, where MCMC is run for a sufficient number of iterations to eliminate dependence on the starting point; (2) estimation of the target parameter (e.g., p-value) using the sum of weights; and (3) update trial function parameter to balance the number of samples from different regions of the sample space. The algorithm is repeated until convergence or maximum number of iterations is achieved.

