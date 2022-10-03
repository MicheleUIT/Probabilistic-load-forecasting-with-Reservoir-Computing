# Notes on MCMC
Source: https://towardsdatascience.com/monte-carlo-markov-chain-mcmc-explained-94e3a6c8de11 %%find better ones?%%

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
A Markov Chain is a stochastic process with descrete time that satisfies the Markov property:
$$
	P(X_{n+1}=k|X_n=k_n,X_{n-1}=k_{n-1},\dots,X_1=k_1)=P(X_{n+1}=k|X_n=k_n)
$$
i.e., the $(n+1)$-th state of the system depends only on th $n$-th one.

### Stationary distributions
Assume that at time $i$ the distribution over the states of the system is $S_i$ (e.g., if $S_1=(0.9,0,0.1)$, it means that the system is more likely to be in state $(1,0,0)$). Assume that the distribution at time step $i+1$ is
$$
	S_{i+1}=S_iQ
$$
where $Q$ is a transition probability.
Keep applying $Q$, we can then reconstruct the whole trajectory, and we may find that it saturates and reaches a stationary distribution such that $S=QS$. 
This happens regardless of the starting state.

## MCMC
Estimating the posterior
$$
	P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$
usually is impossible even if you have the numerator (i.e., likelihood x prior), because the marginal probability $P(B)=\sum P(B|A)P(A)$ can still be intractable.
The idea of MCMC is to estimate the posterior without going through Bayes: by starting from a probability distribution and gradually converging to the posterior.

How to be sure that the Markov chain converges to the wanted probability distribution? We use Detailed Balance Sheet %% how does it work? %%

Pros:
- efficient sampling in high dimension
- solve problems with a large state space
Cons:
- MCMC doesn't perform well in approximating distribution with multi modes
- can be computationally expensive