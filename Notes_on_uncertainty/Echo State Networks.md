# Notes on ESN
Source: [A Review of Designs and Applications of Echo State Networks](https://arxiv.org/pdf/2012.02974.pdf) %% Find a better review %%

**Definition:**
>A ESN consists of an input layer, a recurrent layer, called reservoir, with a large number of sparsely connected neurons, and an output layer. The connection weights of the input layer and the reservoir layer are fixed after initialization, and the output weights are trainable and obtained by solving a linear regression problem.

ESN dynamics:
$$
\begin{align*}
	x_t&=f(W_\text{in}u_t+W_\text{res}x_{t-1}) \\
	y_t&=W_\text{out}x_t
\end{align*}
$$
where $u_t\in U\subseteq \mathbb{R}^D$ is the input at time $t$ of the time series, $x_t\in X\subseteq \mathbb{R}^N$ is the state of the reservoir at time $t$, $y_t\in\mathbb{R}^M$ is the output.

Sensitive hyperparameters:
- $w^\text{in}$, the input-scaling parameter
- $\alpha$, the sparsity parameter of $W_\text{res}$
- $\rho(W_\text{res})$, the spectral radius

## Echo State Property

^a82cd8

Source: [Re-visiting the echo state property](https://www.researchgate.net/publication/230656358_Re-visiting_the_echo_state_property)

Loosely the Echo State Property is: the state of the reservoir asymptotically should not depend anymore on its initial state.

This effect also depends on the input sequence, so we require that, given $x_{t+1}=F(x_t,u_{t+1})$, $F$ is defined on $X\times U$ where $X$ and $U$ are compact sets and $F(x_t,u_{t+1})\in X$ and $u_t\in U$, $\forall t \in\mathbb{Z}$. 
The compactness of $X$ is already granted by using bounded nonlinearity functions $f$. Similarly it is quite common that also $U$ is compact.

Let's first define a *left infinite sequence* as $X^{-\infty}:=\{x^{-\infty}=(\dots,x_{-1},x_0)|x_t\in X\forall t\leq 0\}$. With this we can give a proper definition of ESP:

**Definition**:
>A network $F : X \times U \to X$ (with the compactness condition) has the echo state property with respect to $U$ if for any left infinite input sequence $u^{-\infty} \in U^{-\infty}$ and any two state vector sequences $x^{-\infty}$, $y^{-\infty} \in X^{-\infty}$ compatible with $u^{-\infty}$ (i.e., $x_t=F(x_{t-1},u_t), \forall t\leq 0$), it holds that $x_0 = y_0$.

