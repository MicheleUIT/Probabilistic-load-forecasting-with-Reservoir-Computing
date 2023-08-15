# Probabilistic load forecasting with Reservoir Computing

This repository contains the code used for the paper **Probabilistic load forecasting with Reservoir Computing**.

Authors: Michele Guerra, Simone Scardapane, Filippo Maria Bianchi

Abstract:

>Some applications of deep learning require not only to provide accurate results but also to quantify the amount of confidence in their prediction. The management of an electric power grid is one of these cases: to avoid risky scenarios, decision-makers need both *precise* and *reliable* forecasts of, for example, power loads. For this reason, point forecasts are not enough hence it is necessary to adopt methods that provide an *uncertainty quantification*.    
This work focuses on reservoir computing as the core time series forecasting method, due to its computational efficiency and effectiveness in predicting time series.
While the RC literature mostly focused on point forecasting, this work explores the compatibility of some popular uncertainty quantification methods with the reservoir setting. Both Bayesian and deterministic approaches to uncertainty assessment are evaluated and compared in terms of their prediction accuracy, computational resource efficiency and reliability of the estimated uncertainty, based on a set of carefully chosen performance metrics.

## Conda environment

Before using this repository, you should first install all needed libraries with conda
```
conda env create -f environment.yml
```

## How to use

To reproduce the experiments you need to adjust `experiments.py` and adjust the `config` dictionary accordingly.  
Otherwise, to play around and visualise all the different methods, you can work interactively with the `main_*.ipynb` notebooks.

## Citation

If you use this work in your research, please cite
```

```