Sources:
- https://medium.com/the-artificial-impostor/quantile-regression-part-1-e25bdd8d9d43 (it sucks)
- https://towardsdatascience.com/quantile-regression-ff2343c4a03, introductory
- [Five Things You Should Know about Quantile Regression](https://support.sas.com/resources/papers/proceedings17/SAS0525-2017.pdf)
- [Quantile Regression](https://pubs.aeaweb.org/doi/pdfplus/10.1257/jep.15.4.143)

A regular linear regression is achieved by solving the least square problem, which is equivalent to finding the conditional mean value $E[Y|X]$.

The idea of quantile regression is to find the quantile level $\tau$, i.e., that value of $Y=Q_\tau(Y|X)$ such that $Pr(Y\leq Q_\tau )=\tau$. If we perform it for multiple values of $\tau$, we can have a clearer picture of the conditional distribution than just its mean value.

The standard linear regression model is
$$
	E[y_i]=b_0+\sum_{\alpha=1}^p b_\alpha x_{\alpha i} \qquad i=1,\dots,N
$$
and the $b$'s are found by solving
$$
	\min_{b_0,\dots,b_p}\sum_{i=1}^N (y_i-b_0-\sum_\alpha b_\alpha x_{\alpha i})^2
$$
For quantile regression instead
$$
	Q_\tau(y_i)=b_0(\tau)+\sum_{\alpha=1}^p b_\alpha(\tau) x_{\alpha i} \qquad i=1,\dots,N
$$
$$
	\min_{b_0(\tau),\dots,b_p(\tau)}\sum_{i=1}^N \rho_\tau(y_i-b_0(\tau)-\sum_\alpha b_\alpha(\tau) x_{\alpha i})
$$
where $\rho_\tau$ is the check loss
$$
	\rho_\tau(r)=\tau \max(r,0)+(1-\tau)\max(-r,0)
$$
