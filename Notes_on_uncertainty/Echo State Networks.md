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

## Stochastic search variable selection 

^5ae715

Source: [Deep echo state networks with uncertainty quantificationfor spatio-temporal forecasting](https://onlinelibrary.wiley.com/doi/epdf/10.1002/env.2553)

In this paper they use ESNs to tackle the challenge of performing long-lead forecast on spatio-temporal data, while also producing [[Uncertainty in Deep Learning|uncertainty quantification]].

To quantify uncertainty you can:
- implement Bayesian ESN (have a look at the reference they cite), but it can't work with MCMC (https://link.springer.com/content/pdf/10.1007/978-3-319-23525-7_22.pdf, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6177672)
- use an ensemble of weak learners ESN so to have smaller hidden state dimension (wouldn't this just be epistemic uncertainty though?)

Interesting(?) readings by same author:
- https://www.mdpi.com/1099-4300/21/2/184
- https://arxiv.org/abs/2209.02750
- https://arxiv.org/abs/2206.02218