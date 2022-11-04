# Notes on MCMC
Sources: 
- https://towardsdatascience.com/monte-carlo-markov-chain-mcmc-explained-94e3a6c8de11 
- https://cims.nyu.edu/~holmes/teaching/asa19/handout_Lecture3_2019.pdf#page=13&zoom=100,89,628 
- [MCMC using Hamiltonian dynamics](https://arxiv.org/pdf/1206.1901.pdf)
- [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf)
- [Animation of MCMC](https://chi-feng.github.io/mcmc-demo/app.html)
- %%find better ones?%%

MCMC is a sampling technique used to estimate some characteristics of a population, It's composed of [[Monte Carlo Markov Chain#^a1d63b|Monte Carlo]] and [[Monte Carlo Markov Chain#^f90134|Markov Chains]].

## Monte Carlo

^a1d63b

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

^f90134

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

Notice that if $X$ is a Markov chain with distribution $S$ that satisfies detailed balance, then $S$ is stationary.  This is good because solving [[Monte Carlo Markov Chain#^f94fe6|the equation above]] is simpler than directly searching for a stationary distribution.

If the chain satisfies detailed balance, then $Q$ can be symmetrized and its spectrum gives information on the time scales dynamics of the chain (see more [here](https://cims.nyu.edu/~holmes/teaching/asa19/handout_Lecture3_2019.pdf#page=13&zoom=100,89,628)).


## MCMC

^6e6df4

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
- MCMC doesn't perform well in approximating multimodal distributions
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
You can show that the chain satisfies detailed balance wrt $\pi$, i.e., $\pi_iQ_{ij}=\pi_jQ_{ji}$. (Notice that you don't need to know $\pi$, knowing likelihood x prior is enough.)

Notes:
- the chain takes some time before reaching the stationary distribution, it's called *burn-in time*;
- different definitions of the acceptance probability lead to different MCMC;
- the algorithm can be extended to continuous state space;
- a measure of the performance of MCMC is the **correlation time** of a statistic, say $\langle f(x)\rangle_\pi$, $$ \tau_f:=\frac{1}{C_f(0)}\sum_{t=-\infty}^{+\infty}\,C_f(t)dt, \quad \text{where} \quad C_f(t)=\mathbb{E}_\pi[f(X_t)f(X_0)]-(\mathbb{E}[f(X_t)])^2. $$
(I'm not sure about this definition: shouldn't $C_f$ be the correlation between $t$ and 0? What is dt?)

## Hamiltonian MC

^87ebef
^8975bf
^229702
With MH algorithm one performs a random walk in the entire state space and then accept or reject the state according to the mechanism above. This can be very slow in exploring the whole state space. The idea of HMC is to define a Hamiltonian function, which depends on the probability distribution we wish to sample from, and build trajectories in the state space that allows to explore states more efficiently.

### Hamiltonian dynamics
In Hamiltonian dynamics the state is described by two variables: "position" $q$ and "momentum" $p$. The Hamiltonian function corresponds to the "energy" of the system, i.e., kinetic $K$ plus potential energy $U$. In our scenario, the position corresponds to the variables we are interested in, we introduce a corresponding artificial momentum, and the energy is
$$
	H(q,p)=U(q)+K(p)
$$
where $K(p)=p^TM^{-1}p/2$, (with $M$ a symmetric, positive-definite "mass"), which corresponds to $-\log(\text{Gaus}(0,M))$, and the potential $-\log f(q)$, with $f(q)$ the target distribution we wish to sample from.
The equations of motion are
$$
\begin{align*}
	\frac{d q_i}{dt} &= \frac{\partial H}{\partial p_i} = [M^{-1}p]_i \\
	\frac{d p_i}{dt} &= -\frac{\partial H}{\partial q_i} = -\frac{\partial U}{\partial q_i}.
\end{align*}
$$

### Properties
- Hamiltonian dynamics is _reversible_, an this implies that MCMC updates following the dynamics leave the target distribution invariant %%(why?)%%
- The Hamiltonian is invariant (energy is conserved), this implies that the acceptance probability of an update is 1 if $H$ is kept invariant
- It preserves volume in the state space, so in an MCMC we don't need to take into account a possible change in volume when computing the acceptance probability
- Simplecticness

### Leapfrog method
Hamilton's equations need to be discretized to implement them in an algorithm, using a small time step $\varepsilon$.
The leapfrog method is the following (assuming diagonal $M$)
$$
\begin{align*}
	p_i(t+\varepsilon/2)&=p_i(t)-(\varepsilon/2)\frac{\partial U}{\partial q_i}(q(t)) \\
	q_i(t+\varepsilon)&=q_i(t)+\varepsilon p_i(t+\varepsilon/2)/m_i \\
	p_i(t+\varepsilon)&=p_i(t+\varepsilon/2)-(\varepsilon/2)\frac{\partial U}{\partial q_i}(q(t+\varepsilon)).
\end{align*}
$$
The leapfrog method preserves volumes (so it avoids solution diverging to infinity or shrinking to 0) and it is also time-reversible.

### MCMC with Hamiltonian dynamics
Using Hamiltonian dynamics to sample from a distribution requires translating the density function for this distribution to a potential energy function and introducing “momentum” variables to go with the original variables of interest (now seen as “position” variables). We can then simulate a Markov chain in which each iteration resamples the momentum and then does a Metropolis update with a proposal found using Hamiltonian dynamics.

The potential energy function for the posterior will be
$$
	U(q)=-\log(\pi(q)L(q|D))
$$
where $\pi(q)$ is the prior of the model parameters, and $L(q|D)$ is the likelihood given data $D$.
The joint density of $q$ and $p$ is
$$
	f(q,p)=\frac{1}{Z}\exp(-U(q)/T)\,exp(-K(p)/T)
$$
where they are independent: $q$ follows the distribution we are interested in, and the distribution of $p$  depends on our choice of kinetic energy. A possible choice is $K(p)=\sum_i\frac{p_i^2}{2m_i}$ (zero-mean Gaussian with all independent components and variance $m_i$).

HMC proceeds in two steps:
1) Sample $p$ from its Gaussian
2) A new state is proposed using Hamiltonian dynamics: from current state $(q,p)$ we take $L$ steps with the leapfrog method with stepsize $\varepsilon$, and we get to $(q^*,p^*)$. The new state is accepted or rejected according to $$ \min[1,\exp(-H(q^*,p^*)+H(q,p))] $$ (the momentum is then resampled).

HMC is ergodic, so it shouldn't get trapped in a small region of the state space, but it can still happen for some choice of $L$ and $\varepsilon$ (for example, with almost-periodic trajectories).

## Diagnostics
(work in progress, read [this](https://link.springer.com/content/pdf/10.1007/978-0-387-71265-9_6.pdf))

### Choosing parameters
Sensitive parameters are:
- Stepsize $\varepsilon$
- Trajectory length $L$
Metrics:
- What is the optimal acceptance rate?
- Autocorrelation

### Things to check
**Convergence**: if the MC converges to the right posterior
**Mixing**: if the MC moves around throughout the whole density once it has converged

They are affected by:
- *Starting values* for the parameters: for certain starting values convergence may be too slow, so also mixing becomes a problem.
- *Shape of the posterior*: examples:
	- If the posterior is *multimodal*, MC may converge fast to one mode, but it could not be able to mix to the others modes. A solution could be to use a proposal density with larger variance, or find a better model that is not affected by multimodality (maybe multimodality raised by ignoring some important variable). 
	- Strong *correlations* between the parameters of the posterior can cause poor convergence and mixing. A solution is centering the data, or reparameterizing the model so to make them uncorrelated, or changing the posterior completely
- Choice of *proposal density* (is it the variational distribution? no, it should be the model itself)