Link: [Uncertainty in Deep Learning](https://www.cs.ox.ac.uk/people/yarin.gal/website/thesis/thesis.pdf)
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

## CRPS metric

^dd3ff3

A metric to measure how far a probability distribution is from true value is the *continuous rank probability score*
$$
	\text{CRPS}(F,x)=\int_{-\infty}^{+\infty}(F(y)-\theta(y-x))^2dy
$$
where $F$ is the cumulative distribution, $\theta$ is Heaviside function and $x$ is the observation, i.e., the true value.


# Bayesian neural network

^5e8409

We search for the distribution of weights that generated the data $p(\omega|X,Y)$. It's approximated by a $q_\theta(\omega)$, which is factorised over the weights (assuming they are independent). Maximizing the ELB has two problems:
- There is a sum over all data points, it doesn't scale with a lot of data. Solution: mini-batch.
- Evaluating the log-likelihood. Solution: Monte Carlo integration (at least to estimate derivatives wrt $\theta$).

## Custom variational distribution

Il problema dello scrivere una guida custom è fare in modo che le variabili latenti siano correlate. Un approccio è quello usato dall'autoguide: usare gaussiane univariate e poi correlarle con la decomposizione di Cholesky (guarda [qui](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4066115) e [qui](https://cs229.stanford.edu/section/gaussians.pdf)).
Se io voglio usare distribuzioni diverse dalla gaussiana, come adattare questo approccio? Si possono mescolare distribuzioni diverse?
Una possibilità è di usare le gaussiane inizialmente, ma poi quando si va a fare il sampling le si trasforma in distribuzioni diverse (guarda [qui](https://stats.stackexchange.com/a/415553), ma si può fare? Una volta che le correlo posso trasformarle? C'è un altro metodo per altre distribuzioni?)
Ma mi serve davvero tutto questo controllo?

## Sparsity inducing varational distributions

^a68cbd

Using as variational distribution a multi-variate Gaussian that assumes correlation between all dimensions consumes too much memory. Consider an MLP where a linear layer alone has a $500\times 30$ parameters (it reduces the input from a dimension of 500 to a dimension of 30), if those $500\times 30 =15000$ parameters $\omega$ are correlated, then the covariance matrix will have a size of $15000\times 15000=225000000$, which is often too large to store in GPU memory.

Let's assume that $\omega\in\mathbf{R}^p$ is distributed as a $p$-variate normal with 0 mean and covariance matrix $\Sigma$, it depends on a smaller number of latent parameters $\phi\in\mathbf{R}^r$, that is we have $\omega=R\phi+\epsilon$, where $R$ is a $p\times r$ matrix with $r<p$, $\phi$ is a $r$-dimensional random variable from a Gaussian distribution with 0 mean and variance $\mathbf{1}_r$, and $\epsilon$ is a $p$-dimensional vector of independently distributed error terms with zero mean and finite variance $\text{var}(\epsilon)=\Psi$. Then we can compute the covariance matrix of $\omega$
$$
\begin{align}
	\text{cov}(\omega)&=\mathbf{E}[(\omega-\mu)(\omega-\mu)^T]=\mathbf{E}[\omega\omega^T]=\\
	&=\mathbf{E}[(R\phi+\epsilon)(R\phi+\epsilon)^T]=\\
	&=\mathbf{E}[R\phi\phi^T R^T+\epsilon\epsilon^T+R\phi\epsilon^T+\epsilon\phi^T R^T]=\\
	&=R\mathbf{E}[\phi\phi^T]R^T+\mathbf{E}[\epsilon\epsilon^T]+R\mathbf{E}[\phi\epsilon^T]+\mathbf{E}[\epsilon\phi^T] R^T=\\
	&=RR^T+\Psi
\end{align}
$$
because $\mathbf{E}[\epsilon\phi^T]=0$. So the covariance can be expressed as the sum of a low-rank matrix $RR^T$ of rank $r$, and a diagonal matrix $\Psi$.