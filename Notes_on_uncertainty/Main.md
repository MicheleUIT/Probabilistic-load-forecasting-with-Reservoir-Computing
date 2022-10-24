# Notes on Reservoir Computing and uncertainty quantification
## Ideas
### Comparative study
1) Add a probabilistic regression on top of the [[Echo State Networks|ESN]].
2) Compare Bayesian methods ([[Monte Carlo Markov Chain|MCMC]] and [[Uncertainty in Deep Learning#^1595b1|Variational Inference]]) and frequentist methods (quantile regression, ...)
	- what other frequentist methods?
	- I need to find out on what ground I should lead the comparison, i.e., what parameters to explore and what metrics (e.g., how to compare the confidence intervals of the different techniques? statistical validity of the interval?)
	- If possible, analyse separately aleatoric and epistemic uncertainty (see [[Uncertainty in Deep Learning#^f25fa8|here]])
3) Compare with fully trainable architectures (https://github.com/manuwhs/BayesianRNN)
4) Address the problem of doing regression on the output of the reservoir
	- It's too large for MCMC
	- With VI there is no problem
	- [Here](https://arxiv.org/abs/1806.10728) they sample the number of regression parameter
	- Other dimensionality reduction techniques, like PCA and laplacian eigenmaps
5) Add [[Accurate uncertainties for Deep Learning using calibrated regression|calibration]] on top and compare the different methods
	- How to properly measure the (lack of) calibration? See [[Accurate uncertainties for Deep Learning using calibrated regression#^ef7dd1|Diagnostic tools]]
	- If the actual coverage of a CI with probability $p$ is less than $p$, shouldn't we also have a measure of the confidence on the coverage so to properly compare them?
6) Apply this on time-series forecasting settings. I would need more datasets if it has to be a comparative study.

## Novel study
Extend the above on graphs, or time-dependant graphs.


## To do list
- [ ] Use a deeper NN
	- [x] find a way to implement more layers
	- [ ] play around with configurations to make it work
- [ ] carefully fix the MCMC so that it works correctly
	- [x] read paper on [[Monte Carlo Markov Chain#^8975bf|HMC]]
	- [ ] read chapter on the diagnostics of HMC
- [ ] explore different dimensionality reduction mechanism (PCA, kPCA)
	- [x] Read about the "[stochastic search variable selection](https://onlinelibrary.wiley.com/doi/epdf/10.1002/env.2553)"
		- [ ] check if there has been further progress since 2018
	- [ ] PCA
		- [ ] does PCA affect the correlation of guide samples?
	- [ ] Laplacian eigenmaps
- [ ] Use frequentist approaches to quantify uncertainties
	- [ ] Read about quantile regression
	- [ ] Use enseble ESN for epistemic uncertainties?
- [ ] Change also priors, likelihoods, posterior:
	- [ ] how to decide?
	- [ ] check other works on Bayesian RNN how they do it
- [ ] Read other papers on calibration
- [ ] Compare raw MCMC with MCMC+SSVS, MCMC+dimensionality reduction (PCA, laplacian eigenmaps, ...)
- [ ] Compare ESN with Bayesian LSTM (check https://github.com/manuwhs/BayesianRNN)
- [ ] Compare also with the [time-series forecaster](https://pyro.ai/examples/forecasting_i.html9) by Pyro (later on)
