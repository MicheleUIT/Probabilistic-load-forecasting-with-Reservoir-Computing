# Notes on ESN
Sources: 
- [A Review of Designs and Applications of Echo State Networks](https://arxiv.org/pdf/2012.02974.pdf) %% Find a better review %%
- [Reservoir computing approaches to recurrent neural network training](https://www.sciencedirect.com/science/article/pii/S1574013709000173) %% Better, but technical%%

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

Let's first define a *left infinite sequence* as $X^{-\infty}:=\{x^{-\infty}=(\dots,x_{-1},x_0)|x_t\in X~\forall t\leq 0\}$. With this we can give a proper definition of ESP:

**Definition**:
>A network $F : X \times U \to X$ (with the compactness condition) has the echo state property with respect to $U$ if for any left infinite input sequence $u^{-\infty} \in U^{-\infty}$ and any two state vector sequences $x^{-\infty}$, $y^{-\infty} \in X^{-\infty}$ compatible with $u^{-\infty}$ (i.e., $x_t=F(x_{t-1},u_t), \forall t\leq 0$), it holds that $x_0 = y_0$.

There are equivalent formulations of ESP, but they are all quite hard to check. 

A first, more practical, way to check ESP is to scale the reservoir matrix $W_{res}$ so that its spectral radius is smaller than 1. It turns out that this condition is neither necessary nor sufficient to ensure ESP.

Counter-example for sufficiency: Consider $U=\{0\}$, so the system is $x_{k+1}=f(Wx_k)$. If $f$ is the identity, then the origin is a global attractor when $\rho(W)<1$, but consider $f(x)=\tanh(x)$ in the 1-dim case. The fixed points solve
$$
	x=\tanh(wx)
$$
if $w>1$ the origin is unstable, so ESP doesn't apply, if $w<1$ then the origin is globally asymptotically stable anyway. So there are no counter-examples in 1-dim. In the above paper they show a counter-example in 2-dim (basically you can have a fixed point other than the origin, and that disproves ESP).

A first sufficient condition, which is too strict, is that $\bar{\sigma}(W)<1$, where  $\bar{\sigma}(W)$ denotes the maximum singular value of $W$.
Another weaker condition uses the concept of Schur stability.

Definition:
>A matrix $W \in \mathbf{R}^{N×N}$ is called _Schur stable_ if there exists a positive definite symmetric matrix $P > 0$ such that $W^TPW − P$ is negative definite. If the matrix $P$ can be chosen as a positive definite diagonal matrix, then $W$ is called diagonally Schur stable. The positive definite and negative definite matrices are denoted by $P > 0$ and $P < 0$, respectively.

Theorem:
>The network given above (no feedback) with internal weight matrix $W$ satisfies the echo state property for any input if $W$ is diagonally Schur stable, i.e. there exists a diagonal $P > 0$ such that $W^TPW − P$ is negative definite.

The diagonal Schur stability is fullfilled for example in the following cases:
- $W=(w_{ij})$ if $\rho(|W|)<1$, where $|W|=(|w_{ij}|)$
- $W$ such that $W_{ij}\geq 0~\forall i,j$ and $\rho(W)<1$
- etc.

The first case above gives a simple recipe to construct a $W$ that satisfies ESP:
1. start with a random $W$ with all non-negative entries $W_{ij}\geq0$
2. scale $W$ so that $\rho(W)<1$
3. change the signs of a desired number of entries to get negative connection weights as well.

It must be noted also that even with a spectral radius larger than 1 is possible to have ESP.