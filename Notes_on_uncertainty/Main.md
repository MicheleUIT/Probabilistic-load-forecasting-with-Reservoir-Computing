# Notes on Reservoir Computing and uncertainty quantification
## Ideas
### Comparative study
1) Add a probabilistic regression on top of the [[Echo State Networks|ESN]].
2) Compare Bayesian methods ([[Markov Chain Monte Carlo|MCMC]] and [[Uncertainty in Deep Learning#^1595b1|Variational Inference]]) and frequentist methods (quantile regression, ...)
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
- [x] Use a deeper NN
	- [x] find a way to implement more layers
	- [x] play around with configurations to make it work
- [x] carefully fix the MCMC so that it works correctly
	- [x] read paper on [[Markov Chain Monte Carlo#^8975bf|HMC]]
	- [x] read chapter on the diagnostics of HMC
- [x] Implement the diagnostic tools for MCMC
	- [x] retrieve acceptance rate
	- [x] retrieve step size
	- [x] plot trace graphs
	- [x] check how the Gelman Rubin factor is computed in Pyro → in pyro.ops.stats
	- [x] calibration
	- [x] autocorrelation
	- [x] effective sample size
- [ ] explore different dimensionality reduction mechanism (PCA, kPCA)
	- [x] [stochastic search variable selection](https://onlinelibrary.wiley.com/doi/epdf/10.1002/env.2553) (SSVS)
		- [x] check if there has been further progress since 2018
		- [x] can we use VI instead of MCMC?
		- [x] as spike-and-slab distribution I use a [[Dimensionality reduction#^cc4054|horseshoe distribution]], be sure that my implementation is correct, it looks different from the original paper
		- [x] how to use a discrete Bernoulli?
	- [ ] PCA
		- [ ] does PCA affect the correlation of guide samples?
	- [ ] Laplacian eigenmaps
- [ ] Use frequentist approaches to quantify uncertainties
	- [x] Read about quantile regression
	- [ ] Use enseble ESN for epistemic uncertainties?
	- [ ] [jackknife+](https://www.stat.cmu.edu/~ryantibs/papers/jackknife.pdf) (it's distribution indipendent)
- [ ] conformal predictions (you don't need to do assumptions on distributions, like  jackknife+, is it frequentist?)
- [x] Implement [[Uncertainty in Deep Learning#^dd3ff3|CRPS]]
- [ ] Change also priors, likelihoods, posterior:
	- [ ] how to decide?
	- [ ] check other works on Bayesian RNN how they do it
- [ ] Read other papers on calibration
- [ ] Compare raw MCMC with MCMC+SSVS, MCMC+dimensionality reduction (PCA, laplacian eigenmaps, ...)
- [ ] Compare ESN with Bayesian LSTM (check https://github.com/manuwhs/BayesianRNN)
- [ ] Compare also with the [time-series forecaster](https://pyro.ai/examples/forecasting_i.html9) by Pyro (later on)
- [ ] Add more datasets
	- [ ] [this](https://github.com/fabridamicelli/kuramoto](https://github.com/fabridamicelli/kuramoto) and [this](https://github.com/gravins/NumGraph](https://github.com/gravins/NumGraph) are synthetic
	- [ ] like [this](https://www.kaggle.com/code/mfaaris/3-ways-to-deal-with-time-series-forecasting)
- [ ] Say something about the advantage of using RC with a Bayesian approach instead of RNN:
	- [ ] would it be possible to use RNN with VI?
	- [ ] would it be possible to use RNN with MCMC? Probably you can't do the backpropagation...
- [ ] Compare models with Bayes factor?
- [ ] Use [Sparse bayesian linear regression](https://pyro.ai/examples/sparse_regression.html)?
