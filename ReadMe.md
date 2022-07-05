# code-Gompertz-Mixture-Approach-for-modeling-COVID-19-dynamic-in-Mexico

Library: It contain all Python modules that we used for all calculations

-opt_baseline: Estimate Gompertz-Mixture model

-opt_baseline_trun: Estimate Gompertz-Mixture model treating with a right truncation

-initial_conditions: Get initial conditions for baseline model, in addition, estimate accelerated model parameters, get Gompertz classic growth model parameters
and obtain the carrying capacity of each process with Levenberg-Marquardt

-samples: This module contains the functions that implement general sampling algorithm and kde sampling algorithm

Images: This folder contains all the generated plots

Data: It contains all data that we can put in a public repository. In one folder, you can find the R script: tail_Gompertz.R. This script adjust Peaks-over
threshold method in order to estimate Gompertz tail and complete data of a required process. 

GMM.ipynb: This is a Jupyter notebook with a summarized explanation of workflow and use of above scripts.
