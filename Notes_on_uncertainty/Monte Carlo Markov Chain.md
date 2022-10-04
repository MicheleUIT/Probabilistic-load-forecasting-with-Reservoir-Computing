# Notes on MCMC
Sources: 
- https://towardsdatascience.com/monte-carlo-markov-chain-mcmc-explained-94e3a6c8de11 
- https://cims.nyu.edu/~holmes/teaching/asa19/handout_Lecture3_2019.pdf#page=13&zoom=100,89,628 
- %%find better ones?%%

MCMC is a sampling technique used to estimate some characteristics of a population, It's composed of [[Monte Carlo Markov Chain^Monte Carlo|Monte Carlo]] and [[Monte Carlo Markov Chain^Markov Chains|Markov Chains]].

## Monte Carlo
Example: Assume you have to estimate an expectation
$$
	s=\int\,p(x)f(x)dx=\mathbb{E}_p[f(x)]
$$
which is too difficult to solve analytically, then we can approximate it by taking a lot of samples of $f(x)$ and compute
$$
	\hat{s}_n=\frac{1}{n}\sum_{i=1}^n\,f(x^{(i)})
$$
Sometimes though it could be hard to sample from $f$, that's when we use Markov Chains.

## Markov Chains
A Markov Chain is a stochastic process $X=\{X_0,X_1,\dots\}$, where $X_i$ is a discrete RV, that satisfies the Markov property:
$$
	P(X_{n+1}=k|X_n=k_n,X_{n-1}=k_{n-1},\dots,X_1=k_1)=P(X_{n+1}=k|X_n=k_n)
$$
i.e., the $(n+1)$-th state of the system depends only on th $n$-th one.

### Stationary distributions
Assume that at time $n$ the distribution over the states of the system is $S_i(n)=P(X_n=i)$ (e.g., if $S(n)=(0.9,0,0.1)$, it means that the system is more likely to be in state $(1,0,0)$). You can write the distribution at time step $n+1$ is
$$
	S(n+1)=S(n)Q
$$
where $Q_{ij}(n)=P(X_{n+1}=j|X_n=i)$ is a transition probability.

A Markov chain is *homogeneous* if the transition probability doesn't depend on time $n$, i.e.,
$$
	P(X_{n+1}=j|X_n=i)=P(X_{1}=j|X_0=i).
$$

Keep applying $Q$, we can then reconstruct the whole trajectory, and we may find that it saturates and reaches a *stationary* distribution such that $S=SQ$, regardless of the starting state.

### Detailed balance

^098190

Let $S$ be a stationary distribution, the chain satisfies *detailed balance* if
$$
	S_iQ_{ij}=S_jQ_{ji} \quad \forall i,j
$$

^f94fe6

**NB**: The stationary condition is $S_i=\sum_j S_jQ_{ji}$, in the detailed balance equation above there is no sum.

Notice that if $X$ is a markov chain with distribution $S$ that satisfies detailed balance, then $S$ is stationary.  This is good because solving [[Monte Carlo Markov Chain#^f94fe6|the equation above]] is simpler than than directly searching for a stationary distribution.

If the chain satisfies detailed balance, then $Q$ can be symmetrized and its spectrum gives information on the time scales dynamics of the chain (see more [here](https://cims.nyu.edu/~holmes/teaching/asa19/handout_Lecture3_2019.pdf#page=13&zoom=100,89,628)).


## MCMC
Estimating the posterior
$$
	P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$
usually is impossible even if you have the numerator (i.e., likelihood x prior), because the marginal probability $P(B)=\sum P(B|A)P(A)$ can still be intractable.
The idea of MCMC is to estimate the posterior without going through Bayes: by starting from a probability distribution and gradually converging to the posterior.

Pros:
- efficient sampling in high dimension
- solve problems with a large state space
- you don't need to compute the normalization constant $P(B)$
Cons:
- MCMC doesn't perform well in approximating distribution with multi modes
- can be computationally expensive

How to be sure that the Markov chain converges to the wanted probability distribution? We use [[Monte Carlo Markov Chain#^098190|detailed balance]].

### Metropolis-Hasting algorithm
Imagine that the target discrete probability distribution is $\pi$. Take any Markov chain whose state space contains the support of $\pi$, with transition matrix $H$ according to the following steps:
1. Suppose we start from $X_n=i$, we want to find $X_{n+1}$
2. Chose proposal state $Y=j$ according to the $i$-th row of $H$, i.e., $H_{ij}=P(Y=j|X_n=i)$
3. Compute the acceptance probability $$a_{ij}=\min\left(1,\frac{\pi_j H_{ji}}{\pi_iH_{ij}}\right)$$
4. Sample $U\sim\text{Uniform}(0,1)$
	1. if $U<a_{ij}$ accept the proposal state, i.e., $X_{n+i}=Y$
	2. if $U>a_{ij}$ reject it, $X_{n+1}=X_n$

The induced Markov chain has transition probability
$$
	Q_{ij}=
	\begin{cases}
		H_{ij}a_{ij} & i\neq j \\
		1-\sum_{i \neq j} H_{ij}a_{ij} & i=j
	\end{cases}
$$
You can show that the chain satisfies detailed balance wrt $\pi$, i.e., $\pi_iQ_{ij}=\pi_jQ_{ji}$.

Notes:
- the chain takes some time before reaching the stationary distribution, it's called *burn-in time*;
- different definitions of the acceptance probability lead to different MCMC;
- the algorithm can be extended to continuous state space;
- a measure of the performance of MCMC is the **correlation time** of a statistic, say $\langle f(x)\rangle_\pi$, $$ \tau_f:=\frac{1}{C_f(0)}\sum_{t=-\infty}^{+\infty}\,C_f(t)dt, \quad \text{where} \quad C_f(t)=\mathbb{E}_\pi[f(X_t)f(X_0)]-(\mathbb{E}[f(X_t)])^2. $$
(I'm not sure about this definition: shouldn't $C_f$ be the correlation between $t$ and 0?)
