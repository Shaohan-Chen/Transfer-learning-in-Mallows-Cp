# Transfer-learning-in-Mallows-Cp
This repository contains the code for reproducing the experimental results of ''Transfer Learning in Information Criteria-based Feature Selection" (https://arxiv.org/abs/2107.02847).
Within the download, you'll find four folders: *TLCp_simulation*, *TLCp_iron_experiments*, *TLCp_school_experiments*, and *TLCp_parkinson_experiments*. Each folder provides the functions and the associate datasets to estimate the regression coefficients using the proposed two TLCp methods and other compared benchmarks.
1. The *TLCp_simulation* folder includes five subfolders:
- *approximate_Cp_data_interaction*. This folder contains functions to calculate the MSE values of the approximate Cp and the original Cp estimators in the presence of feature correlations using the simulated data sets.
- *approximate_TLCp_data_interaction*. This folder contains functions to calculate the MSE values of the approximate TLCp cutoff and the original TLCp estimators and in the presence of feature correlations by the simulated data sets.
- *benchmarks_data_interaction*. This folder includes functions to calculate the MSE values of the aggregate benchmark methods.
- *High-dimensional_simulation*. This folder provides functions to solve the approximate Cp, the approximate TLCp, and the least-squares problems in the high-dimensional setting. The main procedure compares the MSE performance of these methods.
*simulation_expansion*. This folder contains functions to calculate the MSE values of the orthogonal Cp and TLCp estimators under the different combinations of the relative task dissimilarity and the target sample size. We can easily reproduce other simulation results under the orthogonal assumption by modifying this function slightly.
2. The *TLCp_school_experiments* folder includes 11 subfolders:
- *aggregate_LASSO*. This folder contains functions to calculate the unexplained variance of the aggregate LASSO (running the LASSO on the aggregate dataset formed by combining data for both the target and source tasks) using school data sets [^1]. We can easily reproduce the experimental results of the standard LASSO by modifying this function slightly.
- *aggregate_original_Cp*. This folder contains functions to calculate the unexplained variance of the aggregate original Cp criterion using school data sets.
- *aggregate_stepwise*. This folder contains functions to calculate the unexplained variance of the aggregate stepwise feature selection (FS) using school data sets. We can easily reproduce the experimental results of the stepwise FS by modifying this function slightly.
- *aggregate_univariate*. This folder contains functions to calculate the unexplained variance of the aggregate univariate FS using school data sets. We can easily reproduce the experimental results of the univariate FS by modifying this function slightly.
- *approximate_TLCp*. This folder contains functions to calculate the unexplained variance of the approximate TLCp method (with one source task) using school data sets. 
- *approximate_TLCp_multisource*. This folder contains functions to calculate the unexplained variance of the approximate TLCp method (with two source tasks) using school data sets. 
- *multi_L21*. This folder contains functions to calculate the unexplained variance of the least L21-norm method [^1][^2] using school data sets. 
- *multilevel_lasso*. This folder contains functions to calculate the unexplained variance of the multi-level LASSO [^3] using school data sets. 
- *original_Cp*. This folder contains functions to calculate the unexplained variance of the Mallows' Cp criterion using school data sets.
- *original_TLCp*. This folder contains functions to calculate the unexplained variance of the original TLCp method (with one source task) using school data sets.
- *original_TLCp._multisource*. This folder contains functions to calculate the unexplained variance of the original TLCp method (with two source tasks) using school data sets.
3. The illustration for the *TLCp_school_experiments* folder can be applied to the *TLCp_iron_experiments* and *TLCp_parkinson_experiments* folders.
4. This repository is currently only available for MATLAB. The user needs MATLAB with 2021a or higher versions.
[^1]: Zhou, Jiayu, Jianhui Chen, and Jieping Ye. "Malsar: Multi-task learning via structural regularization." Arizona State University 21 (2011).
[^2]: Lounici, Karim, et al. "Taking advantage of sparsity in multi-task learning." arXiv preprint arXiv:0903.1468 (2009).
[^3]: Lozano, Aurelie C., and Grzegorz Swirszcz. "Multi-level lasso for sparse multi-task regression." Proceedings of the 29th International Coference on International Conference on Machine Learning. 2012.
