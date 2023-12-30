# [Probabilistic load forecasting with Reservoir Computing](https://ieeexplore.ieee.org/abstract/document/10360823)

This repository contains the code used for the paper **Probabilistic load forecasting with Reservoir Computing**.

Authors: Michele Guerra, Simone Scardapane, Filippo Maria Bianchi

Abstract:

>Some applications of deep learning require not only to provide accurate results but also to quantify the amount of confidence in their prediction. The management of an electric power grid is one of these cases: to avoid risky scenarios, decision-makers need both *precise* and *reliable* forecasts of, for example, power loads. For this reason, point forecasts are not enough hence it is necessary to adopt methods that provide an *uncertainty quantification*.    
This work focuses on reservoir computing as the core time series forecasting method, due to its computational efficiency and effectiveness in predicting time series.
While the RC literature mostly focused on point forecasting, this work explores the compatibility of some popular uncertainty quantification methods with the reservoir setting. Both Bayesian and deterministic approaches to uncertainty assessment are evaluated and compared in terms of their prediction accuracy, computational resource efficiency and reliability of the estimated uncertainty, based on a set of carefully chosen performance metrics.

## Conda environment

Before using this repository, you should first install all needed libraries with conda
```
conda env create -f conda_environment.yml
```

## How to use

To reproduce the experiments you need to adjust `experiments.py` and adjust the `config` dictionary accordingly.  
Otherwise, to play around and visualise all the different methods, you can work interactively with the `main_*.ipynb` notebooks.

## Citation

If you use this work in your research, please cite
```
@ARTICLE{10360823,
  author={Guerra, Michele and Scardapane, Simone and Bianchi, Filippo Maria},
  journal={IEEE Access}, 
  title={Probabilistic Load Forecasting With Reservoir Computing}, 
  year={2023},
  volume={11},
  number={},
  pages={145989-146002},
  doi={10.1109/ACCESS.2023.3343467}}
```
