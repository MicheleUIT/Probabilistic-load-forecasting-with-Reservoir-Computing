Link: [Uncertainty in Deep Learning](https://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)
Author: Yarin Gal
Code: [GitHub code](https://github.com/yaringal)

# Introduction
In many scenarios it's important to be able to tell how certain we are about a model output: is it making sensible predictions or guessing at random?
If we train a model on a dataset, and then provide a data point that lies outside the training data distribution, the model should give me a prediction, but also tell me its uncertainty/confidence about it: if it's under-confident or falsely over-confident.

### Types of uncertainties:

^f25fa8

- **aleatoric uncertainty**: observed labels could be noisy due to noisy data (e.g., measurements imprecision)
- **model uncertainty**:
	- coming from uncertainty in model parameters (a lot of models could explain the same dataset)
	- uncertainty on the model structure itself

From the above we want to extract *predictive uncertainty*.

### Model uncertainties
Examples to quantify model uncertainties:
- Bayesian neural networks: each weight is described by a probability distribution, but they are often not practical
- Stochastic regularization techniques: for example dropout, where you have a stochastic forward pass that you can use also during inference, not just in training, to have an approximate predictive distribution for each input
$$
	\begin{align*}
		x\to \{y_1,\dots,y_N\} \\
		\mathbb{E}[y]=\frac{1}{N}\sum_i\,y_i(x) \\
		\text{Var}[y]=\mathbb{E}[y^2]-\mathbb{E}[y]^2
	\end{align*}
$$
# Bayesian modelling
Training data: $X=\{x_1,\dots,x_N\}$, and $Y=\{y_1,\dots,y_N\}$
Model: $y=f^\omega(x)$, with parameters $\omega$
Question: What parameters are likely to have generated our data?

Elements:
- *prior* distribution $p(\omega)$, i.e., our prior belief on the parameters
- *likelihood* distribution $p(y|x,\omega)$, which is the probabilistic model; example for classification is the softmax likelihood $p(f^\omega(x)|x,\omega)=\text{softmax}(f^\omega(x))$

Target: *posterior* distribution
$$
	p(\omega|X,Y)=\frac{p(Y|X,\omega)p(\omega)}{p(Y|X)}
$$
*Inference*: predicting the output distribution for a new input $x^*$
$$
	p(y^*|x^*,X,Y)=\int p(y^*|x^*,\omega)p(\omega|X,Y)\,d\omega
$$
Problem: being able to solve analytically $p(Y|X)=\int p(Y|X,\omega)p(\omega)d\omega$ is often impossible.

## Variational inference
You approximate the true posterior with a simpler variational distribution $p(\omega|X,Y)\simeq q_\theta(\omega)$, and minimize the KL divergence w.r.t. $\theta$ 
$$
	\text{KL}(q_\theta(\omega)||p(\omega|X,Y))=\int q_\theta(\omega)\log\frac{q_\theta(\omega)}{p(\omega|X,Y)}d\omega
$$
An equivalent (proved below) approach is maximizing the *evidence lower bound* (ELBO)
$$
	\begin{align*}
	\mathcal{L}_{\text{VI}}(\theta):=\mathcal{L}_\text{LL}-\text{KL}(q_\theta(\omega)||p(\omega)) \\
	\mathcal{L}_\text{LL}=\int q_\theta(\omega)\log p(Y|X,\omega)d\omega
	\end{align*}
$$
where:
- $\mathcal{L}_\text{LL}$ is the expected log-likelihood, and push $q_\theta(\omega)$ to explain data well
- second term pushes $q_\theta(\omega)$ close to the prior
In fact,
$$
\begin{align}
	\text{KL}(q_\theta(\omega)||p(\omega|X,Y))&=\mathbb{E}_q[\log q_\theta(\omega)]-\mathbb{E}_q[\log p(\omega|X,Y)]= \\
	&=\mathbb{E}_q[\log q_\theta(\omega)]-\mathbb{E}_q[\log p(Y|X,\omega)]-\mathbb{E}_q[\log p(\omega)]+\mathbb{E}_q[\log p(Y|X)] = \\
	&\leq \mathbb{E}_q[\log q_\theta(\omega)]-\mathbb{E}_q[\log p(Y|X,\omega)]-\mathbb{E}_q[\log p(\omega)] =\\
	&= \text{KL}(q_\theta(\omega)||p(\omega)) - \mathcal{L}_\text{LL}
\end{align}
$$

>The Bayesian modelling strategies aim at solving an integral (marginalisation). Variational inference becomes an optimization problem, so we compute derivatives (still not the same as in DL, since here we optimise over distributions, not single values).

# Bayesian neural network

^5e8409

We search for the distribution of weights that generated the data $p(\omega|X,Y)$. It's approximated by a $q_\theta(\omega)$, which is factorised over the weights (assuming they are independent). Maximizing the ELB has two problems:
- There is a sum over all data points, it doesn't scale with a lot of data. Solution: mini-batch.
- Evaluating the log-likelihood. Solution: Monte Carlo integration (at least to estimate derivatives wrt $\theta$).