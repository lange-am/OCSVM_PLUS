# All unit tests
#
# Copyright (C) 2021 Andrey M. Lange, 
# Skoltech, https://crei.skoltech.ru/cdise
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at 
# http://www.boost.org/LICENSE_1_0.txt)

import unittest
import numpy as np
import test_stlcache
import ocsvm_plus_debug as ocsvm_plus

LOG = True
DEEP_LOG = False
DEEP_DEEP_LOG = False
CMP_TOL = 1e-10

class TestSTLCache(unittest.TestCase):  # test stlcache library, different cache policies
    def test_lru_lfu(self):
        self.assertTrue(test_stlcache.TEST_LRU_LFU_RES)


def generate2Dnormal(n_samples, corr_coef):
    m = np.array([0, 0])
    Sigma = np.eye(2) + np.diag([corr_coef], 1) + np.diag([corr_coef], -1)  # Sigma = [[1, corr_coef], [corr_coef, 1]]
    return np.random.multivariate_normal(m, Sigma, n_samples)


def assrt(f, res): 
    return False if not f else res


def ad_assertion(ad, res=True, log=False):
    res = assrt(((np.array(ad.a) == 0) == (np.array(ad.ais0) == 1)).all(), res)
    res = assrt(((np.array(ad.a) != 0) == (np.array(ad.ais0) == 0)).all(), res)

    res = assrt(((np.array(ad.d) == 0) == (np.array(ad.dis0) == 1)).all(), res)
    res = assrt(((np.array(ad.d) != 0) == (np.array(ad.dis0) == 0)).all(), res)

    res = assrt(((np.array(ad.d) == 1) == (np.array(ad.dis1) == 1)).all(), res)
    res = assrt(((np.array(ad.d) != 1) == (np.array(ad.dis1) == 0)).all(), res)

    anot0_dnot01 = set()
    anot0_d0 = set()
    anot0_d1 = set()
    a0_dnot01 = set()
    a0_d1 = set()
    a0_d0 = set()
    for i, (a0, d0, d1) in enumerate(zip(ad.ais0, ad.dis0, ad.dis1)):
        if a0 == 0 and d0 == 0 and d1 == 0:
            anot0_dnot01.update([i])
        if a0 == 0 and d0 == 1:
            res = assrt(d1 == 0, res)
            anot0_d0.update([i])
        if a0 == 0 and d1 == 1:
            res = assrt(d0 == 0, res)
            anot0_d1.update([i])
        if a0 == 1 and d0 == 0 and d1 == 0:
            a0_dnot01.update([i])
        if a0 == 1 and d1 == 1:
            res = assrt(d0 == 0, res)
            a0_d1.update([i])
        if a0 == 1 and d0 == 1:
            res = assrt(d1 == 0, res)
            a0_d0.update([i])

    if log:
        print('anot0_dnot01:', ad.anot0_dnot01)
        print('anot0_d0:', ad.anot0_d0) 
        print('anot0_d1:', ad.anot0_d1) 
        print('a0_dnot01:', ad.a0_dnot01)
        print('a0_d1:', ad.a0_d1)
        print('a0_d0:', ad.a0_d0) 
        
    res = assrt(anot0_dnot01 == ad.anot0_dnot01, res)
    res = assrt(anot0_d0 == ad.anot0_d0, res)
    res = assrt(anot0_d1 == ad.anot0_d1, res)
    res = assrt(a0_dnot01 == ad.a0_dnot01, res)
    res = assrt(a0_d1 == ad.a0_d1, res)
    res = assrt(a0_d0 == ad.a0_d0, res)

    # union of all provides the whole set
    res = assrt(anot0_dnot01.union(anot0_d0).union(anot0_d1).union(a0_dnot01).union(a0_d1).union(a0_d0) == set(range(len(ad.a))), res)

    # no mutial intersection
    res = assrt(anot0_dnot01.union(anot0_d0).union(anot0_d1).union(a0_dnot01).union(a0_d1).intersection(a0_d0) == set(), res)
    res = assrt(anot0_dnot01.union(anot0_d0).union(anot0_d1).union(a0_dnot01).union(a0_d0).intersection(a0_d1) == set(), res)
    res = assrt(anot0_dnot01.union(anot0_d0).union(anot0_d1).union(a0_d1).union(a0_d0).intersection(a0_dnot01) == set(), res)
    res = assrt(anot0_dnot01.union(anot0_d0).union(a0_dnot01).union(a0_d1).union(a0_d0).intersection(anot0_d1) == set(), res)
    res = assrt(anot0_dnot01.union(anot0_d1).union(a0_dnot01).union(a0_d1).union(a0_d0).intersection(anot0_d0) == set(), res)
    res = assrt(anot0_d0.union(anot0_d1).union(a0_dnot01).union(a0_d1).union(a0_d0).intersection(anot0_dnot01) == set(), res)

    return res


class TestAlphaDeltas(unittest.TestCase):  # test of alpha- and delta-coeffs manager
    def test_ad(self):
        n_samples = 100
        ad = ocsvm_plus.AlphasDeltas(n_samples)
        res = True

        for i in np.random.choice(range(n_samples), n_samples*n_samples):
            if np.random.choice(range(2), 1) == 0:
                d = np.random.choice(range(3), 1)
                if d == 0:  # delta = 0
                    ad.set_d0(i)
                    if DEEP_LOG:
                        print('d[', i, ']=', 0)
                    res = ad_assertion(ad, res, DEEP_DEEP_LOG)
                elif d == 1:  # delta = 1
                    ad.set_d1(i)
                    if DEEP_LOG: 
                        print('d[', i, ']=', 1)
                    res = ad_assertion(ad, res, DEEP_DEEP_LOG)
                else:
                    # d == 2
                    ad.set_dnot01(i, 0.5)
                    if DEEP_LOG:
                        print('d[', i, ']=', 0.5)
                    res = ad_assertion(ad, res, DEEP_DEEP_LOG)
            else:
                a = np.random.choice(range(2), 1)
                if a == 0:
                    ad.set_a0(i)
                    if DEEP_LOG:
                        print('a[', i, ']=', 0)
                    res = ad_assertion(ad, res, DEEP_DEEP_LOG)
                else:
                    ad.set_anot0(i, 25)
                    if DEEP_LOG:
                        print('a[', i, ']=', 25)
                    res = ad_assertion(ad, res, DEEP_DEEP_LOG)

        res = ad_assertion(ad, res, LOG)
        self.assertTrue(res)


class TestKernelManager(unittest.TestCase):
    def test_kernel_set_manager(self):
        n_samples = 100 
        corr_coeff = 0.5
        X = generate2Dnormal(n_samples, corr_coeff)
        X = X[:, 0]

        cache_size = 100
        ker_sigma = 1e-4
        ker_gamma = 1/ker_sigma**2/2
        K = ocsvm_plus.kernel_rbf(ker_gamma)
        km = ocsvm_plus.KernelSetManager(K, cache_size)

        res = True

        def print_output(log=False):
            if LOG:
                print('was_in_cache:', km.was_in_cache, 
                      'was_not_in_cache:', km.was_not_in_cache, 
                      'xy_calculated:', km.xy_calculated,
                      'kernel calculated:', (km.K.ncalls, K.ncalls),
                      'cache_size:', km.cache_size, 
                      'cache_actual_size:', km.cache_actual_size)

        def test_vec():
            return (km.was_in_cache, km.was_not_in_cache, km.xy_calculated, (km.K.ncalls, K.ncalls), km.cache_size, km.cache_actual_size)

        if LOG:
            print('\njust kernel binary function calculations without using a cache:')
        for ix in range(10):
            for iy in range(10):
                km.get(X[[ix]], X[[iy]])
        print_output(LOG)
        if test_vec() != (0, 0, 100, (100, 100), 100, 0):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\nadd elements not seen by cache, not reaching max size, i.e. cache_size:')
        for ix in range(5):
            for iy in range(5):
                km.get(X[[ix]], X[[iy]], ix, iy)
        print_output(LOG)
        if test_vec() != (10, 15, 100, (115, 115), 100, 15):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\nadd elements both already seen by cache and 5 not seen:')
        for ix in range(6):
            for iy in range(5):
                km.get(X[[ix]], X[[iy]], ix, iy)
        print_output(LOG)
        if test_vec() != (35, 20, 100, (120, 120), 100, 20):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\nadd elements where only one was not seen by cache:')
        for ix in range(6):
            for iy in range(6):
                km.get(X[[ix]], X[[iy]], ix, iy)
        print_output(LOG)
        if test_vec() != (70, 21, 100, (121, 121), 100, 21):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\nadd elements to overreach cache limit:')
        for ix in range(6):
            for iy in range(20):
                km.get(X[[ix]], X[[iy]], ix, iy)
        print_output(LOG)
        if test_vec() != (106, 105, 100, (205, 205), 100, 100):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\nafter reset cache:')
        km.reset()
        for ix in range(2):
            for iy in range(50):
                km.get(X[[ix]], X[[iy]], ix, iy)
        print_output(LOG)
        if test_vec() != (1, 99, 0, (304, 304), 100, 99):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\n2 new elements unseen, then old (0, 0) was exluded and (0, 1) was not:')
        for ix in range(1):
            for iy in range(50, 52):
                km.get(X[[ix]], X[[iy]], ix, iy)
        km.get(X[[0]], X[[0]], 0, 0)
        km.get(X[[0]], X[[1]], 0, 1)
        print_output(LOG)
        if test_vec() != (2, 102, 0, (307, 307), 100, 100):
            res = False
        if LOG:
            print(res)

        self.assertTrue(res)

    def test_kernel_matrix_manager(self):
        n_samples = 100 
        corr_coeff = 0.5
        X = generate2Dnormal(n_samples, corr_coeff)
        X = X[:, 0]

        cache_size = 100
        ker_sigma = 1e-4
        ker_gamma = 1/ker_sigma**2/2
        K = ocsvm_plus.kernel_rbf(ker_gamma)
        km = ocsvm_plus.KernelMatrixManager(K, cache_size)

        res = True

        def print_output(log=False):
            if LOG:
                print('was_in_cache:', km.was_in_cache, 
                      'was_not_in_cache:', km.was_not_in_cache, 
                      'xy_calculated:', km.xy_calculated,
                      'kernel calculated:', (km.K.ncalls, K.ncalls),
                      'cache_size:', km.cache_size, 
                      'cache_actual_size:', km.cache_actual_size)

        def test_vec():
            return (km.was_in_cache, km.was_not_in_cache, km.xy_calculated, (km.K.ncalls, K.ncalls), km.cache_size, km.cache_actual_size)

        if LOG:
            print('\njust kernel binary function calculations without using a cache:')
        for ix in range(10):
            for iy in range(10):
                km.get(X[[ix]], X[[iy]])
        print_output(LOG)
        if test_vec() != (0, 0, 100, (100, 100), 100, 5050):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\nadd elements not seen by cache:')
        for ix in range(10):
            for iy in range(10):
                km.get(X[[ix]], X[[iy]], ix, iy)
        print_output(LOG)
        if test_vec() != (45, 55, 100, (155, 155), 100, 5050):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\nadd 10 new elements and 100 already seen:')
        for ix in range(11):
            for iy in range(10):
                km.get(X[[ix]], X[[iy]], ix, iy)
        print_output(LOG)
        if test_vec() != (145, 65, 100, (165, 165), 100, 5050):
            res = False
        if LOG:
            print(res)

        if LOG:
            print('\nafter reset cache:')
        km.reset()
        for ix in range(20):
            for iy in range(20):
                km.get(X[[ix]], X[[iy]], ix, iy)
        print_output(LOG)
        if test_vec() != (190, 210, 0, (375, 375), 100, 5050):
            res = False
        if LOG:
            print(res)

        self.assertTrue(res)


class TestOneClassSVM_plus(unittest.TestCase):
    def test_delta_pair(self):
        n_samples = 100
        corr_coeff = 0.5
        np.random.seed(0)
        X = generate2Dnormal(n_samples, corr_coeff)

        if LOG:
            logging_file_name = self.__class__.__name__+'.test_delta_pair.log'
        else:
            logging_file_name = None

        model = ocsvm_plus.OneClassSVM_plus(n_features=1, nu=0.5012345, 
                                            alg='delta_pair',
                                            logging_file_name=logging_file_name)
        model.fit(X)

        check =           np.all(model.decision_function(X)   - model.decision_function(X[:, :1])    < CMP_TOL)
        check = check and np.all(model.correcting_function(X) - model.correcting_function(X[:, -1:]) < CMP_TOL)
        check = check and np.any(model.decision_function(X) < 0)        
        check = check and model.fit_status_

        self.assertTrue(model.fit_status_)

    def test_best_step(self):
        n_samples = 100
        corr_coeff = 0.5
        np.random.seed(0)
        X = generate2Dnormal(n_samples, corr_coeff)

        if LOG:
            logging_file_name = self.__class__.__name__+'.test_best_step.log'
        else:
            logging_file_name = None

        model = ocsvm_plus.OneClassSVM_plus(n_features=1, 
                                            alg='best_step',
                                            logging_file_name=logging_file_name)
        model.fit(X)

        check =           np.all(model.decision_function(X)   - model.decision_function(X[:, :1])    < CMP_TOL)
        check = check and np.all(model.correcting_function(X) - model.correcting_function(X[:, -1:]) < CMP_TOL)
        check = check and np.any(model.decision_function(X) < 0)
        check = check and model.fit_status_

        self.assertTrue(model.fit_status_)

    def test_best_step_2d(self):
        n_samples = 100
        corr_coeff = 0.5
        np.random.seed(0)
        X = generate2Dnormal(n_samples, corr_coeff)

        if LOG:
            logging_file_name = self.__class__.__name__+'.test_best_step_2d.log'
        else:
            logging_file_name = None

        model = ocsvm_plus.OneClassSVM_plus(n_features=1,
                                            alg='best_step_2d',
                                            kernel_cache_size = n_samples,
                                            logging_file_name=logging_file_name)
        model.fit(X)
        
        check =           np.all(model.decision_function(X)   - model.decision_function(X[:, :1])    < CMP_TOL)
        check = check and np.all(model.correcting_function(X) - model.correcting_function(X[:, -1:]) < CMP_TOL)
        check = check and np.any(model.decision_function(X) < 0)
        check = check and model.fit_status_

        self.assertTrue(check)

    def test_caches(self):
        # Checks that kernel caching does not affect the output (affects only processing time)
        n_samples = 100
        corr_coeff = 0.5
        np.random.seed(0)
        X = generate2Dnormal(n_samples, corr_coeff)

        if LOG:
            log0 = self.__class__.__name__+'.test_caches0.log'
            log1 = self.__class__.__name__+'.test_caches1.log'
            log2 = self.__class__.__name__+'.test_caches2.log'
        else:
            log0 = None
            log1 = None
            log2 = None

        model0 = ocsvm_plus.OneClassSVM_plus(n_features=1, logging_file_name=log0, random_seed=0).fit(X)


        model1 = ocsvm_plus.OneClassSVM_plus(n_features=1,
                                             kernel_cache_size = n_samples,
                                             logging_file_name=log1, random_seed=0).fit(X)

        model2 = ocsvm_plus.OneClassSVM_plus(n_features=1,
                                             distance_cache_size = n_samples,
                                             logging_file_name=log2, random_seed=0).fit(X)

        check =           np.all(model0.decision_function(X)   - model1.decision_function(X[:, :1])    < CMP_TOL)
        check = check and np.all(model0.alphas_   - model1.alphas_ < CMP_TOL)
        check = check and np.all(model1.alphas_   - model2.alphas_ < CMP_TOL)
        check = check and np.all(model1.correcting_function(X) - model2.correcting_function(X[:, -1:]) < CMP_TOL)
        check = check and np.any(model0.decision_function(X) < 0)
        check = check and model0.fit_status_

        self.assertTrue(check)


if __name__ == '__main__':
    unittest.main()
 