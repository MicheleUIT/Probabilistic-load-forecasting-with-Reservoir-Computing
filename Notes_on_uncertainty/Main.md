# Notes on Reservoir Computing and uncertainty quantification
## Ideas
### Comparative study
1) Add a probabilistic regression on top of the [[Echo State Networks|ESN]].
2) Compare Bayesian methods ([[Monte Carlo Markov Chain|MCMC]] and [[Uncertainty in Deep Learning#^1595b1|Variational Inference]]) and frequentist methods (quantile regression, ...)
	- what other frequentist methods?
	- I need to find out on what ground I should lead the comparison, i.e., what parameters to explore and what metrics (e.g., how to compare the confidence intervals of the different techniques? statistical validity of the interval?)
3) Address the problem of doing regression on the output of the reservoir
	- It's too large for MCMC
	- With VI there is no problem %% why? %%
	- [Here](https://arxiv.org/abs/1806.10728) they sample the number of regression parameter
	- Other dimensionality reduction techniques, like PCA
4) Add [[Accurate uncertainties for Deep Learning using calibrated regression|calibration]] on top and compare the different methods
5) Apply this on time-series forecasting settings. I would need more datasets if it has to be a comparative study.

## Novel study
Extend the above on graphs, or time-dependant graphs.


