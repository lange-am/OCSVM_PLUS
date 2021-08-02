# One-Class SVM+

Unsupervised outlier detection with *privileged information*. Generalizes One-Class *`nu`*-SVM that estimates the support of a high-dimensional distribution by accounting for additional (privileged) set of features available on training stage but unavailable on new data scoring and predicting (for example, future behaviour of a time series). 

# Installation

You can simply download the repository in your folder. Then from its root directory (where `setup.py` is placed) run in the command line

`python setup.py build_ext --inplace`

After this you can import the library from python:

`>>> import ocsvm_plus`

Be sure that Python 3 is used, for this you may need print 'python3' instead of 'python'. You also need cython installed (`pip install Cython` or similar). For more confidence, you should run unittests: 

`python setup_debug.py build_ext --inplace`<br/>
`python -m unittest test_ocsvm_plus`

Now debug version is also available:

`>>> import ocsvm_plus_debug`

Debug version performs assertions, obtains intermediate results in different ways and checks the equivalence, dumps more detailed info to logged file. It works much slower than basic (release) version.

# API

```python
class ocsvm_plus.OCSVM_PLUS(kernel='rbf', kernel_gamma='scale', 
                            kernel_star='rbf', kernel_star_gamma='scale', 
                            nu=0.5, gamma='auto', tau=0.001, 
                            alg='best_step_2d', ff_caches='not_bound', 
                            kernel_cache_size=0, distance_cache_size=0, 
                            max_iter=-1, random_seed=None, 
                            logging_file_name=None)
```

## Parameters

| **Parameters:**                                                                                              |  **Description**                                       |
| :-------                                                                                                     | :-------                                               |
| <strong>kernel: *{'rbf', 'linear'} or a class derived from ocsvm_plus.kernel, defailt='rbf'*</strong>        | Kernel K for original features `X`.                    |
| <strong>kernel_gamma: *{'scale', 'auto'} or float, default='scale'*</strong>                                 | Kernel coefficient if K is 'rbf'. For `kernel_gamma='scale'` (default) it uses `kernel_gamma=1/(X.shape[1] * X.var())` as value of `kernel_gamma`. For 'auto' it uses `kernel_gamma=1/X.shape[1]`. |
| <strong>kernel_star: *{'rbf', 'linear'} or a class derived from ocsvm_plus.kernel, defailt='rbf'*</strong>   | Kernel K* for privileged features `X_star`.            |
| <strong>kernel_star_gamma: *{'scale', 'auto'} or float, default='scale'*</strong>                            | Kernel coefficient if K* is 'rbf'. For `kernel_star_gamma='scale'` (default) it uses `kernel_star_gamma=1/(X_star.shape[1] * X_star.var())` as value of kernel_star_gamma. For 'auto' it uses `kernel_star_gamma=1/X_star.shape[1]`.     |
| <strong>nu: *float, default=0.5*</strong>                                                                    | Parameter of `nu`-SVM, original features regularizer, should be between (0, 1). |
| <strong>gamma: *'auto' or float, default='auto'*</strong>                                                    | Privileged features regularizer.                 |
| <strong>tau: *float, default=0.001*</strong>                                                                 | Tolerance for stopping criterion.                |
| <strong>alg: *{'best_step_2d', 'best_step', 'delta_pair'}, defailt='best_step_2d'*</strong>                  | Mode for optimization procedure, affects processing time. For `'delta_pair'` the algorithm tries to find the most `delta`-violating pair of privileged coefficients and, if no `delta`-pairs remained, then it looks for `alpha`-violating pair of original coefficients. Although `delta`-optimization step for a given `delta`-pair is cheaper in time than `alpha`-step for a pair of `alpha`-coefficients (due to decision and correcting function updates), this option can demonstrate slow convergence. `'best_step'` tries to find both the worst `alpha`- and the worst `delta`-violating pairs and the most violating one is selected for the optimization step. `'best_step_2d'` is similar to `'best_step'`, but if the selected pair is also a violating pair according to another criterion (not necessarily being the worst), then the two-dimensional optimization is applied (for example, if the most violating `alpha`-pair of training examples is also a `delta`-violating pair, then the shift for two alpha coefficients and a shift for two delta coefficients are found together). `'best_step_2d'` and `'best_step'` may demonstrate comparable fastness, because the optimization of two pairs of coefficients at once is at the same time more computationally expensive.|
| <strong>ff_caches: *{'all', 'not_bound', 'not_zero'}, defailt='not_bound'*</strong>                          | Sets ranges for caches of indices of training examples `C` and `C*` for which the values of decision `f(x_i)` and correcting `f*(x*_i)` functions (without their intercepts `–rho` and `b*`) are kept actual on every iteration using simple updates (by subtract old pair of coefficients and add new ones) and without need to full recalculation (summing over all non-zero coefficients). `alpha`-pair is selected over `C`, `delta`-pair is over `C*`. Affects processing time of the optimization procedure. Wide cache provides better choice of violation pairs and less expensive function recalculations, while small cache provides faster violating pair search and smaller number of often `f(x_i)` and `f*(x*_i)` updates. <br /> <br /> `'all'`: both caches include all training examples. `'not_bound'`: `C` includes elements `i` for which `alpha_i>0` or `0<delta_i<1`, `C*` includes all elements. `'not_zero'`: `C` and `C*` covers all elements except those who has both `alpha_i` and `delta_i` zero. Generally, `ff_cache` is a dict with keys `'anot0_dnot01'`, `'anot0_d0'`, `'anot0_d1'`, `'a0_dnot01'`, `'a0_d1'`, `'a0_d0'`, corresponding to subsets of elements (respectively, `alpha_i>0`, `0<delta_i<1`; `alpha_i>0`, `delta_i=0`; `alpha_i>0`, `delta_i=1`; `alpha_i=0`, `0<delta_i<1`; `alpha_i=0`, `delta_i=1`; `alpha_i=0`, `delta_i=0`) and key values: 2 – subset set belongs to `C` and `C*`, or 1-belongs only to `C*`, or 0-not in caches (`all` is equivalent to all key values are 2, `not_zero` - all are 2 except `ff_cache ['a0_d0']=0`, `'not_bound'` is equivalent to `ff_caches = {'anot0_dnot01': 2, 'anot0_d0': 2, 'anot0_d1': 2, 'a0_dnot01': 2, 'a0_d1': 1, 'a0_d0': 1}`).|
| <strong>kernel_cache_size: *int, defailt=0*</strong>                                                         | Size of a cache (number of elements) to store K and K* values according to LRU policy. If 0, then the  kernels K(x_i, x_j) and K*(x*_i, x*_j) are calculated only once and stored as ij-elements of triangular matrices. Limited cache size is memory-efficient, while 0 is the most time-efficient setting. |
| <strong>distance_cache_size: *int, defailt=0*</strong>                                                       | Size of a cache (number of elements) to store values (K_ii-2K_ij+K_jj)/(nu * n_samples) and (K*_ii-2K*_ij+K*_jj)/gamma, LRU policy is used. If 0, then once a value is calculated, it is stored in triangular matrix. Limited cache size is memory-efficient, while 0 is the most time-efficient setting.|
| <strong>max_iter: *int, defailt=-1*</strong>                                                               | Hard limit on iterations within solver, or -1 for no limit. |
| <strong>random_seed: *int or None, defailt=None*</strong>                                                  | Random generator initialization. |
| <strong>logging_file_name: *str or None, defailt=None*</strong>                                            | Text file to dump intermediate results of iterative process of model training. If None, then no logging is performed. |

## Attributes

| **Attributes:**    | **Description**               |
| :-------           | :-------                      |
| **alphas_**        | Dual coefficients `alpha_i`     |
| **deltas_**        | Dual coefficients `delta_i`     |
| **rho_**           | Decision function intercept   |
| **b_star_**        | Correcting function intercept |
| **alpha_support_** | Indices `i` of training examples such that `alpha_i>0`. |
| **delta_support_** | Indices `i` of training examples such that `0<delta_i<1`. |
| **fit_status_**    | Returns `True` if there were enough support vectors to find intercepts `rho` and `b_star`, False otherwise. If False, one should make `tau` smaller.| 

## Methods

|**Methods:**                  |**Description**|
| :-------                     | :-------      |
|`fit(Xall[, y=None])`         | Detects the soft boundary of the set of original vectors `X[n_samples, n_features]` accounting for privileged vectors `X_star[n_samples, n_features_star]`, where `Xall=[n_samples, n_features + n_features_star]` comprises `X` in the first `n_features` columns and `X_star` in the last `n_features_star` columns. `Xall` is array-like, if not C-ordered contiguous array it is copied. Returns `self`.|
|`decision_function(X)`        | Signed distance to the separating hyperplane in original feature space, positive for an inlier and negative for an outlier points. `X` is array-like, first `n_features` columns are used, others are ignored. Returns `ndarray` of shape `(X.shape[0], )`. |
|`correcting_function(X_star)` | Slack variables modelled using privileged features. `X_star` is array-like, last `n_features_star` columns are used as privileged vectors, others are ignored. Returns `ndarray` of shape `(X_star.shape[0], )`. |
|`predict(X)`                 | Perform classification on samples in `X`. Similar to `decision_function(X)`, but the output is +1 if the decision function is positive and -1 otherwise.|

## Examples

## Third party software
STLCACHE library https://github.com/akashihi/stlcache is used for caching the values of kernel functions, many thanks to the authors.
