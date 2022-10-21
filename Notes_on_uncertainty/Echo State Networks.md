# Notes on ESN
Source: [A Review of Designs and Applications of Echo State Networks](https://arxiv.org/pdf/2012.02974.pdf) %% Find a better review %%

**Definition:**
>A ESN consists of an input layer, a recurrent layer, called reservoir, with a large number of sparsely connected neurons, and an output layer. The connection weights of the input layer and the reservoir layer are fixed after initialization, and the output weights are trainable and obtained by solving a linear regression problem.

ESN dynamics:
$$
\begin{align*}
	x(t)&=f(W_\text{in}u(t)+W_\text{res}x(t-1)) \\
	y(t)&=W_\text{out}x(t)
\end{align*}
$$
where $u(t)\in \mathbb{R}^D$ is the input at time $t$ of the time series, $x(t)\in \mathbb{R}^N$ is the state of the reservoir at time $t$, $y(t)\in\mathbb{R}^M$ is the output.

Sensitive hyperparameters:
- $w^\text{in}$, the input-scaling parameter
- $\alpha$, the sparsity parameter of $W_\text{res}$
- $\rho(W_\text{res})$, the spectral radius

Echo State Property: The state of the reservoir should asymptotically depend only on the driving input signal.

# Dimensionality reduction

In order to couple MCMC with ESN one needs to reduce the dimensionality of the reservoir, otherwise the state space for the MCMC to explore would be too big.
^5ae715

Source: [Deep echo state networks with uncertainty quantificationfor spatio-temporal forecasting](https://onlinelibrary.wiley.com/doi/epdf/10.1002/env.2553)

In this paper they use ESNs to tackle the challenge of performing long-lead forecast on spatio-temporal data, while also producing [[Uncertainty in Deep Learning|uncertainty quantification]].

To quantify uncertainty you can:
- implement Bayesian ESN (have a look at the reference they cite, but they are not with MCMC, so what are they? VI?) (https://link.springer.com/content/pdf/10.1007/978-3-319-23525-7_22.pdf -> this could be just a regular RNN, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6177672)
- use an ensemble of weak learners ESN so to have smaller hidden state dimension (wouldn't this just be epistemic uncertainty though?)

In the D-EESN algorithm, they represent the dimension reduction step with $\mathcal{Q}(h)$, where $\mathcal{Q}$ is either PCA or _laplacian eigenmaps_. The problem with this is that it doesn't take fully in account all sources of uncertainty, that's why they switch to a Bayesian approach (MCMC).
In the Bayesian approach (BD-EESN) they sample an ensemble of reservoirs and then perform a Bayesian regression. The regression parameters are then shrinked with a stochastic variable selection (SSVS) prior, that is, many regression parameters are set to zero or close to zero.

## Stochastic search variable selection 

^1f53d5

Source: [Approaches for Bayesian variable selection](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n26.pdf)

In a general regression setting, the variable selection problem is the following: given a variable $Y$ and a set of $p$ potential regressors $X_1,\dots,X_p$, we want to find the best model $Y=X_1^*\beta_1+\cdots+X_q^*\beta_q+\epsilon$, where $X_1^*,\dots,X_q^*$ is a subset of $X_1,\dots,X_p$.

The naive approach would be to calculate the posterior probabilities of all possible $2^p$ models. The idea of Bayesian variable selection is that high probability models are more likely to appear quickly when exploring all possibilities.

More formally, the regression problem would give a posterior of the form
$$
	f(Y|\beta,\sigma)=N_n(X\beta,\sigma^2\mathbb{I})
$$
where $X=(X_1,\dots,X_p)$ is a $n\times p$ matrix and some of the $\beta$'s might be negligible (we want to know which one). We denote a particular choice among the $2^p$ with a vector $\gamma=(\gamma_1,\dots,\gamma_p)$, where $\gamma_i=0$ if $\beta_i$ is small, otherwise $\gamma_i=1$. The appropriate value of $\gamma$ is unknown, so we model it with a prior $\pi(\beta,\sigma,\gamma)=\pi(\beta|\sigma,\gamma)\pi(\sigma|\gamma)\pi(\gamma)$. Different choices of these priors lead to different variable selection strategies. For example, for the $\gamma$
$$
	\pi(\gamma)=\prod w_i^{\gamma_i}(1-w_i)^{1-\gamma_i}.
$$
The posterior distribution $\pi(\gamma|Y)$ contains the relevant information for variable selection: given data $Y$, those models with higher $\pi(\gamma|Y)$ are those supported the most by data and the chosen priors. On a practical level:
- priors must be chosen so that the final posterior makes sense (it is actually higher for more interesting models)
- it should be possible to compute at least where $\pi(\gamma|Y)$ gets higher values.

To explore $\pi(\gamma|Y)$ one can use [[Monte Carlo Markov Chain#^6e6df4|MCMC]] (like Gibbs sampler or HMC). The idea is that, even if $p$ is large, the $\gamma$ values of interest, i.e., those with higher probability, will appear more frequently and quicker. SSVS uses Gibbs sampler.

## Laplacian eigenmaps

Source: [Laplacian eigenmaps and spectral techniques for embedding and clustering](https://proceedings.neurips.cc/paper/2001/file/f106b7f99d2cb30c3db1c3cc0fde9ccb-Paper.pdf)


Interesting(?) readings by same author:
- https://www.mdpi.com/1099-4300/21/2/184
- https://arxiv.org/abs/2209.02750
- https://arxiv.org/abs/2206.02218