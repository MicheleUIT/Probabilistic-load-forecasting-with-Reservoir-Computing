Link: [Accurate uncertainties for Deep Learning using calibrated regression](https://arxiv.org/pdf/1807.00263.pdf)
Authors: Volodymyr Kuleshov, Nathan Fenner, Stefano Ermon

## Introduction

Faithfully estimating model [[Uncertainty in Deep Learning |uncertainty]] can be as important as getting high accuracy.
Using [[Uncertainty in Deep Learning#^5e8409|BNN]] uncertainty estimates are often miscalibrated.

Here they extend a post-processing technique, previously introduced for classification tasks, to calibrate the output of regression.
The idea is to train a "calibrating" model $R:[0,1]\to[0,1]$ on top of a pre-trained classifier so that $R\circ H$ is calibrated.

**Notation**: Within the paper, they use $\mathbb{I}\{\dots\}$ to indicate the "pythonic" casting from a bool to a integer, so it's 1 if the condition within the parentheses is satisfied, otherwise it's 0. See [[Accurate uncertainties for Deep Learning using calibrated regression#^6faa0e|comment]] to understand how it translates for time-series.

## Calibrated classification
Consider binary classification, with data $x_t,y_t\in\mathcal{X}\times\mathcal{Y}$, where $\mathcal{Y}=\{0,1\}$. A forecaster is a function $H$ that outputs a probability distribution, $H:x_t\mapsto F_t(y)$. If the forecaster is calibrated, we expect that if it assigns a probability of 80% to an event, than that event should occur 80% of the time (i.e., for about 80% of the samples $\{x_t\}$, since those are i.i.d.). In formula:
$$
	\frac{\sum_{t=1}^T y_t\mathbb{I}\{H(x_t)=p\}}{\sum_{t=1}^T \mathbb{I}\{H(x_t)=p\}} \to p \quad \forall p\in [0,1]
$$
which, although it has a terrible notation, simply means that the number of times that $H$ forecasts a probability of say 80% of $y_t$ being 1, and it actually turns out to be 1, should amount to 80% of the population (in the limit of large $T$).

## Calibrated regression
A similar concept can be introduced for a regression task. So $\mathcal{Y}=\mathbb{R}$ and the forecaster outputs at each time $t$ a CDF $F_t(y)=\mathbb{P}(Y_t\leq y)$ (e.g., in the form of samples from the posterior $\{y_{t,i}\}_{i=1,\dots,N}$), or $F_t^{-1}(p)=\text{inf}\{y:p\leq F_t(y)\}$, so the formula above becomes
$$
	\frac{\sum_{t=1}^T \mathbb{I}\{y_t\leq F_t^{-1}(p)\}}{T}\to p \quad \forall p\in[0,1]
$$
As it is written, it doesn't make much sense for our setting. They are fitting $T$ points $(x_t,y_t)$. Instead for time-series at each time $t$ we have a set of samples $\{y_{t,i}\}_{i=1,\dots,N}$ so at the numerator we should have an average $\sum_i \mathbb{I}\{y_{t,i}\leq F_t^{-1}(p)\}/N$, and then the average over $t$. (Same for calibration.) Maybe they include this average in the $\mathbb{I}$ notation itself. ^6faa0e

## Diagnostic tools

^ef7dd1

### Calibration
- Chose $m$ confidence levels $0\leq p_1 < \dots < p_m \leq 1$;
- then compute the empirical frequency
$$
	\hat{p}_j = \frac{|\{y_t|F_t(y_t)\leq p_j,t=1,\dots,T\}|}{T}
$$
on a diagnostic dataset, different from the training dataset of the BNN and the calibration dataset;
- plot points $\{(p_j,\hat{p}_j)\}_{j=1,\dots,M}$.

If calibrated, the final graph should be a straight line.
As a measure of the quality of the calibration we can use
$$
	\text{cal} = \sum_{j=1}^m w_j(p_j-\hat{p}_j)^2,
$$
where $w_j$ could be 1, or could give a smaller weight to those confidence levels that are harder to calibrate, e.g.,
$$
	w_j \propto |\{y_t|F_t(y_t)\leq p_j,t=1,\dots,T\}|
$$
but in this way you can't compare the error if the sizes of the datasets are different (?).

### Sharpness
A measure of sharpness of the forecast could be simply the variance
$$
	\text{sha} = \frac{1}{T}\sum_{t=1}^{T}\text{var}(F_t).
$$
