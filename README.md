# OCSVM_PLUS

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

| **Parameters:**                                                                                              |  **Description**                                 |
| :-------                                                                                                     | :-------                                         |
| <strong>kernel: *{'rbf', 'linear'} or a class derived from ocsvm_plus.kernel, defailt='rbf'*</strong>        | Kernel K for original features X.                |
| <strong>kernel_gamma: *{'scale', 'auto'} or float, default='scale'*</strong>                               | Kernel coefficient if K is 'rbf'. For kernel_gamma='scale' (default) it uses kernel_gamma=1 / (X.shape[1] * X.var()) as value of kernel_gamma. For 'auto' it uses kernel_gamma=1/X.shape[1]. |
| <strong>kernel_star: *{'rbf', 'linear'} or a class derived from ocsvm_plus.kernel, defailt='rbf'*</strong>   | Kernel K* for privileged features X*.            |
| <strong>kernel_star_gamma: *{'scale', 'auto'} or float, default='scale'*</strong>                            | Kernel coefficient if K* is 'rbf'. For kernel_star_gamma='scale' (default) it uses kernel_star_gamma=1 / (X*.shape[1] * X*.var()) as value of kernel_star_gamma. For 'auto' it uses kernel_star_gamma=1/X*.shape[1]. |
| <strong>nu: *float, default=0.5*</strong>                                                                    | Parameter of `nu`-SVM, original features regularizer, should be between (0, 1). |
| <strong>gamma: *'auto' or float, default='auto'*</strong>                                                    | Privileged features regularizer.                 |
| <strong>tau**: *float, default=0.001*</strong>                                                               | Tolerance for stopping criterion.                |
| <strong>alg: *{'best_step_2d', 'best_step', 'delta_pair'}, defailt='best_step_2d'*</strong>                  | Mode for optimization procedure. For `'delta_pair'` the algorithm tries to find the most `delta`-violating pair of privileged coefficients and, if no `delta`-pairs remained, then it looks for `alpha`-violating pair of original coefficients. Although `delta`-optimization step for a given `delta`-pair is cheaper in time than `alpha`-step for a pair of `alpha`-coefficients (due to decision and correcting function updates), this option can demonstrate slow convergence. `'best_step'` tries to find both the worst `alpha`- and the worst `delta`-violating pairs and the most violating one is selected for the optimization step. `'best_step_2d'` is similar to `'best_step'`, but if the selected pair is also a violating pair according to another criterion (not necessarily being the worst), then the two-dimensional optimization is applied (for example, if the most violating `alpha`-pair of training examples is also a `beta`-violating pair, then the shift for two alpha coefficients and a shift for two delta coefficients are found together). `'best_step_2d'` and `'best_step'` may demonstrate comparable fastness, because the optimization of two pairs of coefficients at once is at the same time more computationally expensive.|
| <strong>ff_caches: *{'all', 'not_bound', 'not_zero'}, defailt='not_bound'*</strong>                          | A policy for caches  |
| <strong>kernel_cache_size: *int, defailt=0*</strong>                                                         | Size of a cache (number of elements) to store K and K* values according to LRU policy. If 0, then the  kernels K(x_i, x_j) and K*(x*_i, x*_j) are calculated only once and stored as ij-elements of triangular matrices. Limited cache size is memory-efficient, while 0 is the most time-efficient setting. |
| <strong>distance_cache_size: *int, defailt=0*</strong>                                                       | Size of a cache (number of elements) to store values (K_ii-2K_ij+K_jj)/(nu * n_samples) and (K*_ii-2K*_ij+K*_jj)/gamma, LRU policy is used. If 0, then once a value is calculated, it is stored in triangular matrix. Limited cache size is memory-efficient, while 0 is the most time-efficient setting.|
| <strong>max_iter: *int, defailt=-1*</strong>                                                               | Hard limit on iterations within solver, or -1 for no limit. |
| <strong>random_seed: *int or None, defailt=None*</strong>                                                  | Random generator initialization. |
| <strong>logging_file_name: *str or None, defailt=None*</strong>                                            | Text file to dump intermediate results of iterative process of model training. If None, then no logging is performed. |


| **Attributes:**  | **Description**                                 |
| :-------         | :-------                                        |
| **kernel**: *{'rbf', 'linear'}, defailt='rbf'*   | Here's this     |


<dl>
  <dd><strong>Parameters:</strong></dd>
    <dd></strong>kernel: </strong> {'rbf', 'linear'}, defailt='rbf'</dd>
  <dd><strong>Attributes:</strong></dd>
  <dd>This is one definition of the second term. </dd>
  <dd>This is another definition of the second term. </dd>
</dl>

