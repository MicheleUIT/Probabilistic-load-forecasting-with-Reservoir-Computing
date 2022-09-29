Link: [Accurate uncertainties for Deep Learning using calibrated regression](https://arxiv.org/pdf/1807.00263.pdf)
Authors: Volodymyr Kuleshov, Nathan Fenner, Stefano Ermon

Faithfully estimating model [[Uncertainty in Deep Learning |uncertainty]] can be as important as getting high accuracy.
Using [[Uncertainty in Deep Learning#^5e8409|BNN]] uncertainty estimates are often miscalibrated.

Here they extend a post-processing technique, previously introduced for classification tasks, to calibrate the output of regression.
The idea is to train a "calibrating" model $R:[0,1]\to[0,1]$ on top of a pre-trained classifier so that $R\circ H$ is calibrated.