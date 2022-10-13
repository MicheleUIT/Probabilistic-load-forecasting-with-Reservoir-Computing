Link: [Accurate uncertainties for Deep Learning using calibrated regression](https://arxiv.org/pdf/1807.00263.pdf)
Authors: Volodymyr Kuleshov, Nathan Fenner, Stefano Ermon

## Introduction

Faithfully estimating model [[Uncertainty in Deep Learning |uncertainty]] can be as important as getting high accuracy.
Using [[Uncertainty in Deep Learning#^5e8409|BNN]] uncertainty estimates are often miscalibrated.

Here they extend a post-processing technique, previously introduced for classification tasks, to calibrate the output of regression.
The idea is to train a "calibrating" model $R:[0,1]\to[0,1]$ on top of a pre-trained classifier so that $R\circ H$ is calibrated.

**Notation**: Within the paper, they use $\mathbb{I}\{\dots\}$ to indicate the "pythonic" casting from a bool to a integer, so it's 1 if the condition within the parentheses is satisfied, otherwise it's 0.

## Calibrated classification
Consider binary classification, with data $x_t,y_t\in\mathcal{X}\times\mathcal{Y}$, where $\mathcal{Y}=\{0,1\}$. A forecaster is a function $H$ that outputs a probability distribution, $H:x_t\mapsto F_t(y)$. If the forecaster is calibrated, we expect that if it assigns a probability of 80% to an event, than that event should occur 80% of the time (i.e., for about 80% of the samples $\{x_t\}$, since those are i.i.d.). In formula:
$$
	\frac{\sum_{t=1}^T y_t\mathbb{I}\{H(x_t)=p\}}{\sum_{t=1}^T \mathbb{I}\{H(x_t)=p\}} \to p \quad \forall p\in [0,1]
$$
which, although it has a terrible notation, simply means that the number of times that $H$ forecasts a probability of say 80% of $y_t$ being 1, and it actually turns out to be 1, should amount to 80% of the population (in the limit of large $T$).

## Calibrated regression
A similar concept can be introduced for a regression cast. So $\mathcal{Y}=\mathbb{R}$ and the forecaster outputs a CDF $F_t(y)=\mathbb{P}(y_t<y)$, so the formula above becomes
$$
	\frac{\sum_{t=1}^T \mathbb{I}\{F_t(y_t)\leq p\}}{T}\to p \quad \forall p\in[0,1]
$$
(re-write it better)