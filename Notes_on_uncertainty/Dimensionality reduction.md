# Dimensionality reduction

^67e5c3

In order to couple MCMC with [[Echo State Networks|ESN]] one needs to reduce the dimensionality of the reservoir, otherwise the state space for the MCMC to explore would be too big.

Source: [Deep echo state networks with uncertainty quantificationfor spatio-temporal forecasting](https://onlinelibrary.wiley.com/doi/epdf/10.1002/env.2553)

In this paper they use ESNs to tackle the challenge of performing long-lead forecast on spatio-temporal data, while also producing [[Uncertainty in Deep Learning|uncertainty quantification]].

To quantify uncertainty you can:
- implement Bayesian ESN (have a look at the reference they cite, but they are not with MCMC, so what are they? VI?) (https://link.springer.com/content/pdf/10.1007/978-3-319-23525-7_22.pdf -> this could be just a regular RNN, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6177672)
- use an ensemble of weak learners ESN so to have smaller hidden state dimension (wouldn't this just be epistemic uncertainty though?)

In the D-EESN algorithm, they represent the dimension reduction step with $\mathcal{Q}(h)$, where $\mathcal{Q}$ is either PCA or _laplacian eigenmaps_. The problem with this is that it doesn't take fully in account all sources of uncertainty, that's why they switch to a Bayesian approach (MCMC).
In the Bayesian approach (BD-EESN) they sample an ensemble of reservoirs and then perform a Bayesian regression. The regression parameters are then shrinked with a stochastic variable selection (SSVS) prior, that is, many regression parameters are set to zero or close to zero.

Interesting(?) readings by same author:
- https://www.mdpi.com/1099-4300/21/2/184 -> Bayesian LSTM
  > Depending on the application, any number of methods [of dimensionality reduction] can be selected from linear methods such as wavelets, splines, or principal components, or nonlinear methods such as Laplacian eigenmaps [41], restricted Boltzmann machines [42], or diffusion maps [43]
- https://arxiv.org/abs/2206.02218

## Variable selection 

^452ca4

### Stochastic search variable selection 

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

Can we use SVI?

### Shrinkage priors

^cc4054

Source: [Bayesian statistics and modelling](https://sci-hub.se/https://doi.org/10.1038/s43586-020-00001-2)

A class of priors $\pi(\beta)$ that one can use is the **spike-and-slab prior**: a mixture of two distributions, one is peaked in zero (spike), which identifies the irrelevant components, and the other is a diffuse distribution (slab), that capture the non-zero coefficients.

I wasn't able to implement the original SSVS, with a Bernoulli, in Pyro, so as spike-and-slab distribution I used a [horseshoe distribution](https://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf) (but I should check if my implementation is correct).

## Laplacian eigenmaps

Source: [Laplacian eigenmaps and spectral techniques for embedding and clustering](https://proceedings.neurips.cc/paper/2001/file/f106b7f99d2cb30c3db1c3cc0fde9ccb-Paper.pdf)


## PCA
