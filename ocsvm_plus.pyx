# OCSVM+: One-Class SVM with Privileged Information model
#
# Copyright (C) 2021 Andrey M. Lange, 
# Skoltech, https://crei.skoltech.ru/cdise
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at 
# http://www.boost.org/LICENSE_1_0.txt)

# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, nonecheck=False, initializedcheck=False
# distutils: language = c++

include 'debug'

import cython
from cython.operator cimport dereference as deref, preincrement as inc

cimport numpy as cnp
import numpy as np
import logging

from libc.math cimport exp as cexp, isnan
from libcpp.set cimport set as cset
from libcpp.pair cimport pair as cpair
from libcpp.vector cimport vector as cvector

cimport stlcache

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t
ctypedef cnp.uint8_t BTYPE_t
DTYPE = np.float64
ITYPE = np.intc
BTYPE = np.uint8
cdef DTYPE_t NP_NAN = np.nan
cdef DTYPE_t DTYPE_CMP_TOL = 1e-10

ctypedef cpair[Py_ssize_t, Py_ssize_t] PAIR_IJ
ctypedef cpair[Py_ssize_t, DTYPE_t] PAIR_IF
ctypedef cpair[PAIR_IJ, DTYPE_t] PAIR_IJF
ctypedef cpair[PAIR_IJF, PAIR_IJF] PAIR_IJF_IJF
ctypedef cset[Py_ssize_t] SET_K
ctypedef cset[Py_ssize_t].iterator SET_K_ITERATOR

ctypedef stlcache.policy_lru KERNEL_CACHE_POLICY_TYPE

ctypedef enum argmin_argmax: argmin, argmax
ctypedef enum alpha_delta: alpha, delta

cdef int NONE_ELEM_LEVEL = 3
cdef int C_CACHE_LEVEL = 2
cdef int C_STAR_CACHE_LEVEL = 1
cdef int ALL_ELEM_LEVEL = 0

cdef int ALG_DELTA_STEP_PREF = 0
cdef int ALG_BEST_STEP = 1
cdef int ALG_BEST_STEP_2D = 2

cdef class kernel:
    cdef public int ncalls

    def __init__(kernel self):
        self.ncalls = 0

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef DTYPE_t get_c(self, DTYPE_t* x, DTYPE_t* y,  Py_ssize_t n_features,
                       Py_ssize_t ix=-1, Py_ssize_t iy=-1): # ix, iy are used in derived classes
        self.ncalls +=1

    IF DEBUG:
        def get(self, DTYPE_t[::1] x, DTYPE_t[::1] y):
            cdef DTYPE_t* x0 = &x[0]
            cdef DTYPE_t* y0 = &y[0]
            return self.get_c(x0, y0, x.shape[0])


cdef class kernel_rbf(kernel):
    cdef DTYPE_t kernel_gamma

    def __init__(kernel_rbf self, DTYPE_t kernel_gamma):
        super().__init__()
        self.kernel_gamma = kernel_gamma

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef DTYPE_t get_c(self, DTYPE_t* x, DTYPE_t* y, Py_ssize_t n_features,
                       Py_ssize_t ix=-1, Py_ssize_t iy=-1): # ix, iy are used in derived classes:
        kernel.get_c(self, x, y, ix, iy)
        cdef Py_ssize_t i
        cdef DTYPE_t d, s = 0.0
        for i in range(n_features):
            d = x[i] - y[i]
            s += d*d
        return cexp(- self.kernel_gamma * s)


cdef class kernel_linear(kernel):
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef DTYPE_t get_c(self, DTYPE_t* x, DTYPE_t* y, Py_ssize_t n_features,
                       Py_ssize_t ix=-1, Py_ssize_t iy=-1): # ix, iy are used in derived classes:
        kernel.get_c(self, x, y, ix, iy) # ix, iy are used in derived classes
        cdef Py_ssize_t i
        cdef DTYPE_t s = 0.0
        for i in range(n_features):
            s += x[i]*y[i]
        return s


cdef class kernel_ii_ij_jj(kernel):
    cdef public kernel K
    cdef readonly DTYPE_t norm_const

    def __init__(self, kernel K,      # kernel or derived KernelManager
                 DTYPE_t norm_const): # nu_nsamples or gamma
        super().__init__()
        self.K = K
        self.norm_const = norm_const

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef DTYPE_t get_c(self, DTYPE_t* x, DTYPE_t* y, Py_ssize_t n_features,
                       Py_ssize_t ix=-1, Py_ssize_t iy=-1):
        return (self.K.get_c(x, x, n_features, ix, ix) - 2*self.K.get_c(x, y, n_features, ix, iy) + self.K.get_c(y, y, n_features, iy, iy)) / self.norm_const

cdef class KernelManager(kernel):
    cdef public kernel K
    cdef readonly Py_ssize_t cache_size
    cdef readonly int was_in_cache
    cdef readonly int was_not_in_cache
    cdef readonly int xy_calculated # number of direct calls by vectors x, y without check in cache, i.e. ix < 0 or iy < 0 passed to .get()

    def __init__(self, kernel K, Py_ssize_t cache_size):
        super().__init__()
        self.K = K
        self.cache_size = cache_size

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef DTYPE_t get_c(self, DTYPE_t* x, DTYPE_t* y, Py_ssize_t n_features, Py_ssize_t ix=-1, Py_ssize_t iy=-1):
        pass

    IF DEBUG:
        def get(self, DTYPE_t[::1] x, DTYPE_t[::1] y, ix=-1, iy=-1):
            cdef DTYPE_t* x0 = &x[0]
            cdef DTYPE_t* y0 = &y[0]
            return self.get_c(x0, y0, x.shape[0], ix, iy)

    cdef reset_c(self):
        self.was_in_cache = 0
        self.was_not_in_cache = 0
        self.xy_calculated = 0

    IF DEBUG:
        def reset(self):
            self.reset_c()

cdef class KernelSetManager(KernelManager):
    cdef stlcache.cache[PAIR_IJ, DTYPE_t, KERNEL_CACHE_POLICY_TYPE] *C

    def __cinit__(self, kernel K, Py_ssize_t cache_size):
        super().__init__(K, cache_size)
        self.C = NULL
        self.reset_c()

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef DTYPE_t get_c(self, DTYPE_t* x, DTYPE_t* y, Py_ssize_t n_features, Py_ssize_t ix=-1, Py_ssize_t iy=-1):
        cdef DTYPE_t* x0
        cdef DTYPE_t* y0
        cdef PAIR_IJ ixiy
        cdef cpair[bint, DTYPE_t] cache_response
        cdef DTYPE_t ker

        if ix < 0 or iy < 0:
            self.xy_calculated += 1
            return self.K.get_c(x, y, n_features, ix, iy)
        else:
            if ix < iy:
                ixiy = PAIR_IJ(ix, iy)
                x0 = x
                y0 = y
            else:
                ixiy = PAIR_IJ(iy, ix)
                x0 = y
                y0 = x

            cache_response = self.C.recorded_find(ixiy)
            if cache_response.first:
                self.was_in_cache += 1
                return cache_response.second
            else:
                ker = self.K.get_c(x0, y0, n_features, ixiy.first, ixiy.second)
                self.C.insert(ixiy, ker)
                self.was_not_in_cache += 1
                return ker

    cdef reset_c(self):
        KernelManager.reset_c(self)
        if self.C != NULL:
            del self.C
        self.C = new stlcache.cache[PAIR_IJ, DTYPE_t, KERNEL_CACHE_POLICY_TYPE](self.cache_size)

    @property
    def cache_actual_size(self):
        return self.C.size()

    def __dealloc__(self):
        del self.C

cdef class KernelMatrixManager(KernelManager):
    cdef cvector[cvector[DTYPE_t]] C

    def __init__(self, kernel K, Py_ssize_t cache_size):
        cdef cvector[DTYPE_t] tmp_vec
        cdef Py_ssize_t i

        super().__init__(K, cache_size)
        for i in range(cache_size):
            tmp_vec.push_back(NP_NAN)
            self.C.push_back(tmp_vec)
        self.reset_c()

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef DTYPE_t get_c(self, DTYPE_t* x, DTYPE_t* y, Py_ssize_t n_features, Py_ssize_t ix=-1, Py_ssize_t iy=-1):
        cdef DTYPE_t* x0
        cdef DTYPE_t* y0
        cdef Py_ssize_t ix0, iy0
        cdef DTYPE_t ker

        if ix < 0 or iy < 0:
            self.xy_calculated += 1
            return self.K.get_c(x, y, n_features, ix, iy)
        else:
            if ix > iy:
                ix0 = ix
                iy0 = iy
                x0 = x
                y0 = y
            else:
                ix0 = iy
                iy0 = ix
                x0 = y
                y0 = x

            ker = self.C[ix0][iy0]
            if not isnan(ker):
                self.was_in_cache += 1
                return ker
            else:
                ker = self.K.get_c(x0, y0, n_features, ix0, iy0)
                self.C[ix0][iy0] = ker
                self.was_not_in_cache += 1
                return ker

    cdef reset_c(self):
        cdef Py_ssize_t i, j

        KernelManager.reset_c(self)
        for i in range(self.cache_size):
            for j in range(i+1):
                self.C[i][j] = NP_NAN

    @property
    def cache_actual_size(self):
        return ((self.cache_size+1) * self.cache_size) // 2


cdef class AlphasDeltas:
    cdef readonly int n_samples
    cdef readonly DTYPE_t[::1] a, d
    cdef readonly BTYPE_t[::1] ais0, dis0, dis1
    cdef readonly SET_K anot0_dnot01, anot0_d0, anot0_d1, a0_dnot01, a0_d1, a0_d0 # - all sets 

    def __init__(self, Py_ssize_t n_samples):
        self.n_samples = n_samples
        self.a = np.zeros(n_samples, dtype=DTYPE) 
        self.d = np.zeros(n_samples, dtype=DTYPE) 
        self.ais0 = np.ones(n_samples, dtype=BTYPE)
        self.dis0 = np.ones(n_samples, dtype=BTYPE)
        self.dis1 = np.zeros(n_samples, dtype=BTYPE)
        self.anot0_dnot01 = []
        self.anot0_d0 =[]
        self.anot0_d1 = []
        self.a0_dnot01 = []
        self.a0_d1 = []
        self.a0_d0 = range(n_samples)

    cdef void set_anot0_c(self, Py_ssize_t i, DTYPE_t v):
        cdef BTYPE_t* p = &self.ais0[i]
        if p[0] == 1:  # alpha 0 ->  
            if self.dis1[i] == 0:
                if self.dis0[i] == 0:
                    self.a0_dnot01.erase(i)
                    self.anot0_dnot01.insert(i)
                else:
                    self.anot0_d0.insert(i)
                    self.a0_d0.erase(i)
            else:
                self.a0_d1.erase(i)
                self.anot0_d1.insert(i)
            p[0] = 0
        self.a[i] = v

    IF DEBUG:
        def set_anot0(self, i, v):
            self.set_anot0_c(i, v)

    cdef void set_a0_c(self, Py_ssize_t i):
        cdef BTYPE_t* p = &self.ais0[i]
        if p[0] == 0:  # alpha -> 0
            if self.dis1[i] == 0:
                if self.dis0[i] == 0:
                    self.anot0_dnot01.erase(i)
                    self.a0_dnot01.insert(i)
                else:
                    self.anot0_d0.erase(i)
                    self.a0_d0.insert(i)
            else:
                self.anot0_d1.erase(i)
                self.a0_d1.insert(i)
            p[0] = 1
            self.a[i] = 0

    IF DEBUG:
        def set_a0(self, i):
            self.set_a0_c(i)

    cdef void set_dnot01_c(self, Py_ssize_t i, DTYPE_t v):
        cdef BTYPE_t* p0 = &self.dis0[i]
        cdef BTYPE_t* p1 = &self.dis1[i]

        if p0[0] == 1: #    # delta 0 -> not bound
            if self.ais0[i] == 1:
                self.a0_dnot01.insert(i)
                self.a0_d0.erase(i)
            else:
                self.anot0_d0.erase(i)
                self.anot0_dnot01.insert(i)
            p0[0] = 0
        elif p1[0] == 1: #  # delta 1 -> not bound
            if self.ais0[i] == 1:
                self.a0_d1.erase(i)
                self.a0_dnot01.insert(i)
            else:
                self.anot0_d1.erase(i)
                self.anot0_dnot01.insert(i)
            p1[0] = 0
        self.d[i] = v


    IF DEBUG:
        def set_dnot01(self, i, v):
            self.set_dnot01_c(i, v)

    cdef void set_d0_c(self, Py_ssize_t i):
        cdef BTYPE_t* p0 = &self.dis0[i]
        cdef BTYPE_t* p1 = &self.dis1[i]
        if p0[0] == 0:     # delta -> 0
            if p1[0] == 1: # delta 1 -> 0
                if self.ais0[i] == 0:
                    self.anot0_d1.erase(i)
                    self.anot0_d0.insert(i)
                else:
                    self.a0_d1.erase(i)
                    self.a0_d0.insert(i)
                p1[0] = 0
            else:          # delta not bound -> 0
                if self.ais0[i] == 0:
                    self.anot0_dnot01.erase(i)
                    self.anot0_d0.insert(i)
                else:
                    self.a0_dnot01.erase(i)
                    self.a0_d0.insert(i)
            p0[0] = 1
            self.d[i] = 0

    IF DEBUG:
        def set_d0(self, i):
            self.set_d0_c(i)

    cdef void set_d1_c(self, Py_ssize_t i):
        cdef BTYPE_t *p0 = &self.dis0[i]
        cdef BTYPE_t *p1 = &self.dis1[i]
        if p1[0] == 0:      # delta -> 1
            if p0[0] == 1:  # delta 0 -> 1
                if self.ais0[i] == 0:
                    self.anot0_d0.erase(i)
                    self.anot0_d1.insert(i)
                else:
                    self.a0_d1.insert(i)
                    self.a0_d0.erase(i)
                p0[0] = 0
            else:           # delta not bound -> 1
                if self.ais0[i] == 0:
                    self.anot0_dnot01.erase(i)
                    self.anot0_d1.insert(i)
                else:
                    self.a0_dnot01.erase(i)
                    self.a0_d1.insert(i)
            p1[0] = 1
            self.d[i] = 1

    IF DEBUG:
        def set_d1(self, i):
            self.set_d1_c(i)

    def a_not0_size(self):
        return self.anot0_dnot01.size() + self.anot0_d0.size() + self.anot0_d1.size()

    def d_not01_size(self):
        return self.anot0_dnot01.size() + self.a0_dnot01.size()

    def d_not0_size(self):
        return self.anot0_dnot01.size() + self.a0_dnot01.size() + self.anot0_d1.size() + self.a0_d1.size()

    if DEBUG:
        def check_coeffs(self):
            for k in self.anot0_dnot01:
                assert(not self.ais0[k] and not self.dis0[k] and not self.dis1[k])
                assert(self.a[k] > 0.0 and self.d[k] > 0.0 and self.d[k] < 1.0)
            for k in self.anot0_d0:
                assert(not self.ais0[k] and self.dis0[k] and not self.dis1[k])
                assert(self.a[k] > 0.0 and self.d[k] == 0.0 and self.d[k] < 1.0)
            for k in self.anot0_d1:
                assert(not self.ais0[k] and not self.dis0[k] and self.dis1[k])
                assert(self.a[k] > 0.0 and self.d[k] > 0.0 and self.d[k] == 1.0)
            for k in self.a0_dnot01:
                assert(self.ais0[k] and not self.dis0[k] and not self.dis1[k])
                assert(self.a[k] == 0.0 and self.d[k] > 0.0 and self.d[k] < 1.0)
            for k in self.a0_d1:
                assert(self.ais0[k] and not self.dis0[k] and self.dis1[k])                
                assert(self.a[k] == 0.0 and self.d[k] > 0.0 and self.d[k] == 1.0)
            for k in self.a0_d0:
                assert(self.ais0[k] and self.dis0[k] and not self.dis1[k])
                assert(self.a[k] == 0.0 and self.d[k] == 0.0 and self.d[k] < 1.0)


        def describe(self, deep=False):
            print('anot0_dnot01 size:', self.anot0_dnot01.size())
            print('anot0_d0 size:', self.anot0_d0.size())
            print('anot0_d1 size:', self.anot0_d1.size())
            print('a0_dnot01 size:', self.a0_dnot01.size())
            print('a0_d1 size:', self.a0_d1.size())
            print('a0_d0 size:', self.a0_d0.size())
            if deep:
                print('a:', list(self.a))
                print('d:', list(self.d))


cdef class OCSVM_PLUS_C:
    cdef readonly DTYPE_t nu, gamma, tau
    cdef readonly kernel kerK, kerK_star
    cdef readonly KernelManager kmK, kmK_star
    cdef readonly KernelManager kmKij, kmKij_star
    cdef readonly Py_ssize_t k_cache_size
    cdef readonly Py_ssize_t kii_ij_jj_cache_size
    cdef readonly object random_seed
    cdef readonly Py_ssize_t n_samples
    cdef readonly Py_ssize_t n_features, n_features_star, n_features_total    
    cdef readonly AlphasDeltas coeffs
    cdef readonly DTYPE_t[::1] f
    cdef readonly DTYPE_t[::1] f_star
    cdef DTYPE_t[:, ::1] X
    cdef DTYPE_t nu_nsamples
    cdef readonly int anot0_dnot01
    cdef readonly int anot0_d0
    cdef readonly int anot0_d1
    cdef readonly int a0_dnot01
    cdef readonly int a0_d1
    cdef readonly int a0_d0
    cdef readonly DTYPE_t rho
    cdef readonly DTYPE_t b_star
    cdef bint logging
    cdef str logging_file_name
    cdef int alg
    cdef int last_f_recalculate_level
    cdef int max_iter
    cdef object logger

    def __init__(self, int n_features, DTYPE_t nu, DTYPE_t gamma, DTYPE_t tau, 
                 kernel K = kernel_linear(),
                 kernel K_star = kernel_linear(),
                 object random_seed = None,
                 dict ff_caches = {'anot0_dnot01': 2, 'anot0_d0': 2, 'anot0_d1': 2, 'a0_dnot01': 2, 'a0_d1': 1, 'a0_d0': 1},
                 Py_ssize_t k_cache_size=0,            # 0: the whole matrix is cached, >0 - set cache is used
                 Py_ssize_t kii_ij_jj_cache_size=0,    # 0: the whole matrix is cached, >0 - set cache is used
                 int alg = ALG_BEST_STEP,
                 max_iter = -1,
                 logging_file_name=None):
        if nu <= 0.0 or nu >= 1.0:
            raise ValueError("must be 0<\nu <1!")
        self.n_features = n_features
        self.nu = nu
        self.gamma = gamma
        self.tau = tau
        self.random_seed = random_seed
        self.kerK = K
        self.kerK_star = K_star
        self.kmK = None
        self.kmK_star = None
        self.kmKij = None
        self.kmKij_star = None
        self.anot0_dnot01 = ff_caches['anot0_dnot01']
        self.anot0_d0 = ff_caches['anot0_d0']
        self.anot0_d1 = ff_caches['anot0_d1']
        self.a0_dnot01 = ff_caches['a0_dnot01']
        self.a0_d1 = ff_caches['a0_d1']
        self.a0_d0 = ff_caches['a0_d0']
        self.k_cache_size = k_cache_size
        self.kii_ij_jj_cache_size = kii_ij_jj_cache_size
        self.f = None
        self.f_star = None
        self.X = None
        self.n_samples = 0
        self.n_features_star = 0
        self.n_features_total = 0
        self.nu_nsamples = 0
        self.coeffs = None
        self.rho = NP_NAN
        self.b_star = NP_NAN
        self.alg = alg
        self.max_iter = max_iter
        self.last_f_recalculate_level = NONE_ELEM_LEVEL

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if logging_file_name is not None:
            self.logging = True
            self.logging_file_name = logging_file_name
            handler = logging.FileHandler(self.logging_file_name, 'w', 'utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            self.logger=logging.getLogger()
            for h in self.logger.handlers:
                h.close()
                self.logger.removeHandler(h)
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
        else:
            self.logging = False
            self.logger = None
            self.logging_file_name = self.__class__.__name__

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)   # turn off negative index wrapping for entire function
    cdef initialize_c(self, cnp.ndarray[DTYPE_t, ndim=2] X):
        IF DEBUG:
            if self.logging:
                logging.debug('initialize_c')

        self.X = X
        self.n_samples = X.shape[0]
        self.nu_nsamples = self.nu * self.n_samples

        self.n_features_total = X.shape[1]
        self.n_features_star = self.n_features_total - self.n_features
        if self.n_features_star < 1:
            raise ValueError("not enough features in X!")

        if self.k_cache_size:
            self.kmK = KernelSetManager(self.kerK, self.k_cache_size)
            self.kmK_star = KernelSetManager(self.kerK_star, self.k_cache_size)
        else:
            self.kmK = KernelMatrixManager(self.kerK, self.n_samples)
            self.kmK_star = KernelMatrixManager(self.kerK_star, self.n_samples)

        if self.kii_ij_jj_cache_size:
            self.kmKij = KernelSetManager(kernel_ii_ij_jj(self.kerK, self.nu_nsamples), self.kii_ij_jj_cache_size)
            self.kmKij_star = KernelSetManager(kernel_ii_ij_jj(self.kerK_star, self.gamma), self.kii_ij_jj_cache_size)
        else:
            self.kmKij = KernelMatrixManager(kernel_ii_ij_jj(self.kerK, self.nu_nsamples), self.n_samples)
            self.kmKij_star = KernelMatrixManager(kernel_ii_ij_jj(self.kerK_star, self.gamma), self.n_samples)

        self.coeffs = AlphasDeltas(self.n_samples)
        cdef int[::1] ii_a
        cdef int[::1] ii_d
        cdef int l
        if self.nu_nsamples.is_integer():
            l = int(self.nu_nsamples)
            ii_a = np.random.choice(range(self.n_samples), l, replace=False).astype(ITYPE)
            ii_d = np.random.choice(range(self.n_samples), l, replace=False).astype(ITYPE)
        else:
            l = int(self.nu_nsamples) + 1
            ii_a = np.random.choice(range(self.n_samples), l, replace=False).astype(ITYPE)
            ii_d = np.random.choice(range(self.n_samples), l, replace=False).astype(ITYPE)
            self.coeffs.set_anot0_c(ii_a[l-1], self.nu_nsamples-int(self.nu_nsamples))
            self.coeffs.set_dnot01_c(ii_d[l-1], self.nu_nsamples-int(self.nu_nsamples))
            ii_a, ii_d = ii_a[:l-1], ii_d[:l-1]

        cdef Py_ssize_t i
        for i in range(ii_a.shape[0]):
            self.coeffs.set_anot0_c(ii_a[i], 1.0)
        for i in range(ii_d.shape[0]):
            self.coeffs.set_d1_c(ii_d[i])

        self.initialize_ff_c()

        self.rho = NP_NAN
        self.b_star = NP_NAN

    cdef recalculate_f(self, int ff_cache_level):
        cdef Py_ssize_t k        
        if self.anot0_dnot01 == ff_cache_level:
            for k in self.coeffs.anot0_dnot01:
                self.f[k] = self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)
        if self.anot0_d0 == ff_cache_level:
            for k in self.coeffs.anot0_d0:
                self.f[k] = self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)
        if self.anot0_d1 == ff_cache_level:
            for k in self.coeffs.anot0_d1:
                self.f[k] = self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)
        if self.a0_dnot01 == ff_cache_level:
            for k in self.coeffs.a0_dnot01:
                self.f[k] = self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)
        if self.a0_d1 == ff_cache_level:
            for k in self.coeffs.a0_d1:
                self.f[k] = self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)
        if self.a0_d0 == ff_cache_level:
            for k in self.coeffs.a0_d0:
                self.f[k] = self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)

        self.last_f_recalculate_level = ff_cache_level

    cdef recalculate_f_star(self, int ff_cache_level):
        cdef Py_ssize_t k
        if self.anot0_dnot01 == ff_cache_level:
            for k in self.coeffs.anot0_dnot01:
                self.f_star[k] = self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)
        if self.anot0_d0 == ff_cache_level:
            for k in self.coeffs.anot0_d0:
                self.f_star[k] = self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)
        if self.anot0_d1 == ff_cache_level:
            for k in self.coeffs.anot0_d1:
                self.f_star[k] = self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)
        if self.a0_dnot01 == ff_cache_level:
            for k in self.coeffs.a0_dnot01:
                self.f_star[k] = self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)
        if self.a0_d1 == ff_cache_level:
            for k in self.coeffs.a0_d1:
                self.f_star[k] = self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)
        if self.a0_d0 == ff_cache_level:
            for k in self.coeffs.a0_d0:
                self.f_star[k] = self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)

    cdef initialize_ff_c(self):
        IF DEBUG:
            if self.logging:
                logging.debug('initialize_ff_c')

        cdef Py_ssize_t i
        self.f = np.empty(self.n_samples, dtype=DTYPE)
        self.f_star = np.empty(self.n_samples, dtype=DTYPE)
        for i in range(self.n_samples):
            self.f[i] = NP_NAN            
            self.f_star[i] = NP_NAN

        self.recalculate_f(C_CACHE_LEVEL)
        self.recalculate_f_star(C_CACHE_LEVEL)
        self.recalculate_f_star(C_STAR_CACHE_LEVEL)


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef get_f_c(self, DTYPE_t* x, Py_ssize_t i=-1):
        cdef Py_ssize_t k
        cdef DTYPE_t s = 0.0
        for k in self.coeffs.anot0_d0:
            s += self.coeffs.a[k] * self.kmK.get_c(x, &self.X[0, 0]+k*self.n_features_total, self.n_features, i, k)
        for k in self.coeffs.anot0_dnot01:
            s += self.coeffs.a[k] * self.kmK.get_c(x, &self.X[0, 0]+k*self.n_features_total, self.n_features, i, k)
        for k in self.coeffs.anot0_d1:
            s += self.coeffs.a[k] * self.kmK.get_c(x, &self.X[0, 0]+k*self.n_features_total, self.n_features, i, k)
        return s/self.nu_nsamples


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing. 
    cdef get_f_star_c(self, DTYPE_t* x_star, Py_ssize_t i=-1):
        cdef Py_ssize_t k        
        cdef DTYPE_t s = 0.0
        for k in self.coeffs.anot0_d0:
            s += self.coeffs.a[k]                      * self.kmK_star.get_c(x_star, &self.X[0, 0]+k*self.n_features_total+self.n_features, self.n_features_star, i, k)
        for k in self.coeffs.anot0_dnot01:
            s += (self.coeffs.a[k] - self.coeffs.d[k]) * self.kmK_star.get_c(x_star, &self.X[0, 0]+k*self.n_features_total+self.n_features, self.n_features_star, i, k)
        for k in self.coeffs.anot0_d1:
            s += (self.coeffs.a[k] - 1.0)              * self.kmK_star.get_c(x_star, &self.X[0, 0]+k*self.n_features_total+self.n_features, self.n_features_star, i, k)
        for k in self.coeffs.a0_dnot01:
            s += - self.coeffs.d[k]                    * self.kmK_star.get_c(x_star, &self.X[0, 0]+k*self.n_features_total+self.n_features, self.n_features_star, i, k)
        for k in self.coeffs.a0_d1:
            s += -                                       self.kmK_star.get_c(x_star, &self.X[0, 0]+k*self.n_features_total+self.n_features, self.n_features_star, i, k)
        return s/self.gamma


    cdef PAIR_IF opt_pair_c(self, SET_K s, PAIR_IF k_f_opt, 
                            argmin_argmax amin_amax_flag,    # +1 to find maximums i and m, -1 to find minimums j and n
                            alpha_delta ad_flag):            # +1 for f+f* in alpha-pair,   -1 for f* in delta-pair
        cdef Py_ssize_t k_tmp
        cdef DTYPE_t f_tmp
        cdef SET_K_ITERATOR it = s.begin()
 
        if k_f_opt.first == -1 and it != s.end():
            k_f_opt.first = deref(it)
            if ad_flag == alpha:
                k_f_opt.second = self.f[k_f_opt.first] + self.f_star[k_f_opt.first]
            else:
                k_f_opt.second = self.f_star[k_f_opt.first]
            inc(it)
        while it != s.end():
            k_tmp = deref(it)
            if ad_flag == alpha:
                f_tmp = self.f[k_tmp] + self.f_star[k_tmp]
            else:
                f_tmp = self.f_star[k_tmp]
            if (f_tmp > k_f_opt.second and amin_amax_flag == argmax) or (f_tmp < k_f_opt.second and amin_amax_flag == argmin):
                k_f_opt.first = k_tmp
                k_f_opt.second = f_tmp
            inc(it)
        return k_f_opt

    cdef PAIR_IJ alpha_pair_c(self, int min_ff_cache_level):
        cdef PAIR_IF i_ff_opt = PAIR_IF(-1, NP_NAN)
        cdef PAIR_IF j_ff_opt = PAIR_IF(-1, NP_NAN)

        if self.anot0_dnot01 >= min_ff_cache_level:
            i_ff_opt = self.opt_pair_c(self.coeffs.anot0_dnot01, i_ff_opt, argmax, alpha)
            j_ff_opt = self.opt_pair_c(self.coeffs.anot0_dnot01, j_ff_opt, argmin, alpha)
        if self.anot0_d0 >= min_ff_cache_level:
            i_ff_opt = self.opt_pair_c(self.coeffs.anot0_d0, i_ff_opt, argmax, alpha)
            j_ff_opt = self.opt_pair_c(self.coeffs.anot0_d0, j_ff_opt, argmin, alpha)
        if self.anot0_d1 >= min_ff_cache_level:
            i_ff_opt = self.opt_pair_c(self.coeffs.anot0_d1, i_ff_opt, argmax, alpha)
            j_ff_opt = self.opt_pair_c(self.coeffs.anot0_d1, j_ff_opt, argmin, alpha)
        if self.a0_dnot01 >= min_ff_cache_level:
            j_ff_opt = self.opt_pair_c(self.coeffs.a0_dnot01, j_ff_opt, argmin, alpha)
        if self.a0_d1 >= min_ff_cache_level:
            j_ff_opt = self.opt_pair_c(self.coeffs.a0_d1, j_ff_opt, argmin, alpha)
        if self.a0_d0 >= min_ff_cache_level:
            j_ff_opt = self.opt_pair_c(self.coeffs.a0_d0, j_ff_opt, argmin, alpha)

        IF DEBUG:
            C = []
            if self.anot0_dnot01 >= min_ff_cache_level:
                C += list(self.coeffs.anot0_dnot01)
            if self.anot0_d0 >= min_ff_cache_level:
                C += list(self.coeffs.anot0_d0)
            if self.anot0_d1 >= min_ff_cache_level:
                C += list(self.coeffs.anot0_d1)
            if self.a0_dnot01 >= min_ff_cache_level:
                C += list(self.coeffs.a0_dnot01)
            if self.a0_d1 >= min_ff_cache_level:
                C += list(self.coeffs.a0_d1)
            if self.a0_d0 >= min_ff_cache_level:
                C += list(self.coeffs.a0_d0)

            tmp = [(k, self.f[k] + self.f_star[k]) for k in C if not self.coeffs.ais0[k]]
            assert((i_ff_opt.first == -1) == (len(tmp)==0))
            i_ff_alt = max(tmp, key=lambda x: x[1]) if len(tmp) > 0 else [-1, NP_NAN]

            tmp = [(k, self.f[k] + self.f_star[k]) for k in C]
            assert((j_ff_opt.first == -1) == (len(tmp)==0))
            j_ff_alt = min(tmp, key=lambda x: x[1]) if len(tmp) > 0 else [-1, NP_NAN]

            if i_ff_alt[0] != -1 and i_ff_opt.first != 1:
                assert(abs(i_ff_alt[1] - i_ff_opt.second) < DTYPE_CMP_TOL)

            if j_ff_alt[0] != -1 and j_ff_opt.first != 1:                
                assert(abs(j_ff_alt[1] - j_ff_opt.second) < DTYPE_CMP_TOL)

            if self.logging:
                logging.debug('found: (i, fi+fi^*)='+str(i_ff_opt)+', (j, fj+fj^*)='+str(j_ff_opt)+
                              '; alternative calculation: '+str(i_ff_alt)+' '+str(j_ff_alt))

        return PAIR_IJ(i_ff_opt.first, j_ff_opt.first)

    cdef PAIR_IJ delta_pair_c(self, int min_ff_cache_level):
        cdef PAIR_IF m_f_opt = PAIR_IF(-1, NP_NAN)
        cdef PAIR_IF n_f_opt = PAIR_IF(-1, NP_NAN)

        if self.anot0_dnot01 >= min_ff_cache_level:
            m_f_opt = self.opt_pair_c(self.coeffs.anot0_dnot01, m_f_opt, argmax, delta)
            n_f_opt = self.opt_pair_c(self.coeffs.anot0_dnot01, n_f_opt, argmin, delta)
        if self.anot0_d0 >= min_ff_cache_level:
            m_f_opt = self.opt_pair_c(self.coeffs.anot0_d0, m_f_opt, argmax, delta)
        if self.anot0_d1 >= min_ff_cache_level:
            n_f_opt = self.opt_pair_c(self.coeffs.anot0_d1, n_f_opt, argmin, delta)
        if self.a0_dnot01 >= min_ff_cache_level:
            m_f_opt = self.opt_pair_c(self.coeffs.a0_dnot01, m_f_opt, argmax, delta)
            n_f_opt = self.opt_pair_c(self.coeffs.a0_dnot01, n_f_opt, argmin, delta)
        if self.a0_d1 >= min_ff_cache_level:
            n_f_opt = self.opt_pair_c(self.coeffs.a0_d1, n_f_opt, argmin, delta)
        if self.a0_d0 >= min_ff_cache_level:
            m_f_opt = self.opt_pair_c(self.coeffs.a0_d0, m_f_opt, argmax, delta)

        IF DEBUG:
            C_star = set()
            if self.anot0_dnot01 >= min_ff_cache_level:
                C_star = C_star.union(self.coeffs.anot0_dnot01)
            if self.anot0_d0 >= min_ff_cache_level:
                C_star = C_star.union(self.coeffs.anot0_d0)
            if self.anot0_d1 >= min_ff_cache_level:
                C_star = C_star.union(self.coeffs.anot0_d1)
            if self.a0_dnot01 >= min_ff_cache_level:
                C_star = C_star.union(self.coeffs.a0_dnot01)
            if self.a0_d1 >= min_ff_cache_level:
                C_star = C_star.union(self.coeffs.a0_d1)
            if self.a0_d0 >= min_ff_cache_level:
                C_star = C_star.union(self.coeffs.a0_d0)

            tmp = [(k, self.f_star[k]) for k in C_star if not self.coeffs.dis1[k]]
            assert((m_f_opt.first == -1) == (len(tmp)==0))
            m_f_alt = max(tmp, key=lambda x: x[1]) if len(tmp) > 0 else [-1, NP_NAN]

            tmp = [(k, self.f_star[k]) for k in C_star if not self.coeffs.dis0[k]]
            assert((n_f_opt.first == -1) == (len(tmp)==0))
            n_f_alt = min(tmp, key=lambda x: x[1]) if len(tmp) > 0 else [-1, NP_NAN]

            if m_f_alt[0] != -1 and m_f_opt.first != 1:
                assert(abs(m_f_alt[1] - m_f_opt.second) < DTYPE_CMP_TOL)
            if n_f_alt[0] != -1 and n_f_opt.first != 1:
                assert(abs(n_f_alt[1] - n_f_opt.second) < DTYPE_CMP_TOL)

            if self.logging:
                logging.debug('found: (m, fm^*)='+str(m_f_opt)+', (n, fn^*)='+str(n_f_opt)+
                              '; alternative calculation: '+str(m_f_alt)+' '+str(n_f_alt))

        return PAIR_IJ(m_f_opt.first, n_f_opt.first)


    cdef update_f_c(self, Py_ssize_t i, Py_ssize_t j, Py_ssize_t k, DTYPE_t t):
        cdef DTYPE_t* x_i = &self.X[0, 0]+i*self.n_features_total
        cdef DTYPE_t* x_j = &self.X[0, 0]+j*self.n_features_total
        cdef DTYPE_t* x_k = &self.X[0, 0]+k*self.n_features_total

        cdef DTYPE_t K1 = self.kmK.get_c(x_i, x_k, self.n_features, i, k)
        cdef DTYPE_t K2 = self.kmK.get_c(x_j, x_k, self.n_features, j, k)
        self.f[k] += t * ( K1-K2 ) / self.nu_nsamples


    cdef update_f_star_c(self, Py_ssize_t m, Py_ssize_t n, Py_ssize_t k, DTYPE_t s):
        cdef DTYPE_t* x_m = &self.X[0, 0]+m*self.n_features_total+self.n_features
        cdef DTYPE_t* x_n = &self.X[0, 0]+n*self.n_features_total+self.n_features
        cdef DTYPE_t* x_k = &self.X[0, 0]+k*self.n_features_total+self.n_features

        cdef DTYPE_t K1 = self.kmK_star.get_c(x_m, x_k, self.n_features_star, m, k)
        cdef DTYPE_t K2 = self.kmK_star.get_c(x_n, x_k, self.n_features_star, n, k)
        self.f_star[k] -= s * ( K1-K2 ) / self.gamma


    cdef f_update_c(self, Py_ssize_t i, Py_ssize_t j, DTYPE_t t):
        if self.anot0_dnot01 == C_CACHE_LEVEL:
            for k in self.coeffs.anot0_dnot01:
                self.update_f_c(i, j, k, t)
        if self.anot0_d0 == C_CACHE_LEVEL:
            for k in self.coeffs.anot0_d0:
                self.update_f_c(i, j, k, t)
        if self.anot0_d1 == C_CACHE_LEVEL:
            for k in self.coeffs.anot0_d1:
                self.update_f_c(i, j, k, t)
        if self.a0_dnot01 == C_CACHE_LEVEL:
            for k in self.coeffs.a0_dnot01:
                self.update_f_c(i, j, k, t)
        if self.a0_d1 == C_CACHE_LEVEL:
            for k in self.coeffs.a0_d1:
                self.update_f_c(i, j, k, t)
        if self.a0_d0 == C_CACHE_LEVEL:
            for k in self.coeffs.a0_d0:
                self.update_f_c(i, j, k, t)


    cdef f_star_update_c(self, Py_ssize_t m, Py_ssize_t n, DTYPE_t s):
        if self.anot0_dnot01 >= C_STAR_CACHE_LEVEL:
            for k in self.coeffs.anot0_dnot01:
                self.update_f_star_c(m, n, k, s)
        if self.anot0_d0 >= C_STAR_CACHE_LEVEL:
            for k in self.coeffs.anot0_d0:
                self.update_f_star_c(m, n, k, s)
        if self.anot0_d1 >= C_STAR_CACHE_LEVEL:
            for k in self.coeffs.anot0_d1:
                self.update_f_star_c(m, n, k, s)
        if self.a0_dnot01 >= C_STAR_CACHE_LEVEL:
            for k in self.coeffs.a0_dnot01:
                self.update_f_star_c(m, n, k, s)
        if self.a0_d1 >= C_STAR_CACHE_LEVEL:
            for k in self.coeffs.a0_d1:
                self.update_f_star_c(m, n, k, s)
        if self.a0_d0 >= C_STAR_CACHE_LEVEL:
            for k in self.coeffs.a0_d0:
                self.update_f_star_c(m, n, k, s)


    cdef alpha_step_c(self, Py_ssize_t i, Py_ssize_t j):
        cdef DTYPE_t* x_i = &self.X[0, 0]+i*self.n_features_total
        cdef DTYPE_t* x_j = &self.X[0, 0]+j*self.n_features_total
        cdef DTYPE_t Kij = self.kmKij.get_c(x_i, x_j, self.n_features, i, j)

        x_i = &self.X[0, 0]+i*self.n_features_total+self.n_features
        x_j = &self.X[0, 0]+j*self.n_features_total+self.n_features
        cdef DTYPE_t Kij_star = self.kmKij_star.get_c(x_i, x_j, self.n_features_star, i, j)
        cdef DTYPE_t Delta_f_sum = self.f[i] + self.f_star[i] - self.f[j] - self.f_star[j]
        cdef DTYPE_t t = - Delta_f_sum / (Kij + Kij_star)
        cdef DTYPE_t ai = self.coeffs.a[i]

        IF DEBUG:
            assert(not self.coeffs.ais0[i])
            assert(Delta_f_sum > self.tau)

        if t <= -ai:
            t = -ai
            self.coeffs.set_a0_c(i)
        else:
            self.coeffs.set_anot0_c(i, ai+t)
        self.coeffs.set_anot0_c(j, self.coeffs.a[j]-t)

        IF DEBUG:
            if self.logging:
                logging.debug('sum(alpha_k): '+str(sum([v for v in self.coeffs.a])))

        self.f_update_c(i, j, t)
        self.f_star_update_c(i, j, -t)
        self.last_f_recalculate_level = NONE_ELEM_LEVEL

        IF DEBUG:
            self.check_ff_update_c()

    cdef int get_ff_cache_level(self, Py_ssize_t k):
        if self.coeffs.ais0[k]:
            if self.coeffs.dis0[k]:
                return self.a0_d0
            elif self.coeffs.dis1[k]:
                return self.a0_d1
            else: 
                return self.a0_dnot01
        else:
            if self.coeffs.dis0[k]:
                return self.anot0_d0 
            elif self.coeffs.dis1[k]:
                return self.anot0_d1
            else:
                return self.anot0_dnot01

    cdef delta_step_c(self, Py_ssize_t m, Py_ssize_t n):
        cdef DTYPE_t* x_m = &self.X[0, 0]+m*self.n_features_total + self.n_features
        cdef DTYPE_t* x_n = &self.X[0, 0]+n*self.n_features_total + self.n_features
        cdef DTYPE_t Kmn_star = self.kmKij_star.get_c(x_m, x_n, self.n_features_star, m, n)
        cdef DTYPE_t Delta_f_star = self.f_star[m] - self.f_star[n]
        cdef DTYPE_t s = Delta_f_star / Kmn_star
        cdef DTYPE_t dm = self.coeffs.d[m]
        cdef DTYPE_t dn = self.coeffs.d[n]

        cdef bint m_in_C = self.get_ff_cache_level(m) == C_CACHE_LEVEL
        cdef bint n_in_C = self.get_ff_cache_level(n) == C_CACHE_LEVEL

        IF DEBUG:
            assert(not self.coeffs.dis1[m])
            assert(not self.coeffs.dis0[n])
            assert(Delta_f_star > self.tau)
            # self.coeffs.check_coeffs()

        if 1-dm < dn:
            if s >= 1-dm:
                s = 1-dm
                self.coeffs.set_d1_c(m)
            else:
                self.coeffs.set_dnot01_c(m, dm + s)
            self.coeffs.set_dnot01_c(n, dn - s)
        elif 1-dm > dn:
            if s >= dn:
                s = dn
                self.coeffs.set_d0_c(n)
            else:
                self.coeffs.set_dnot01_c(n, dn - s)
            self.coeffs.set_dnot01_c(m, dm + s)
        else: # 1-dm == dn:
            if s >= dn:
                s = dn
                self.coeffs.set_d1_c(m)
                self.coeffs.set_d0_c(n)
            else:
                self.coeffs.set_dnot01_c(m, dm + s)
                self.coeffs.set_dnot01_c(n, dn - s)

        IF DEBUG:
            if self.logging:
                logging.debug('sum(delta_k): '+str(sum([v for v in self.coeffs.d])))

        self.f_star_update_c(m, n, s)

        # recalcuate f_n, f_m if m or n moved to C cache
        if self.get_ff_cache_level(m) == C_CACHE_LEVEL and not m_in_C:
            self.f[m] = self.get_f_c(&self.X[0, 0]+m*self.n_features_total, m)
        if self.get_ff_cache_level(n) == C_CACHE_LEVEL and not n_in_C:
            self.f[n] = self.get_f_c(&self.X[0, 0]+n*self.n_features_total, n)

        self.last_f_recalculate_level = NONE_ELEM_LEVEL

        IF DEBUG:
            self.check_ff_update_c()
            assert(s*(s*Kmn_star - 2*Delta_f_star) <= 0.0)

    IF DEBUG:
        cdef check_ff_update_c(self):
            cdef Py_ssize_t k
            if self.anot0_dnot01 == C_CACHE_LEVEL:
                for k in self.coeffs.anot0_dnot01:
                    assert(abs(self.f[k] - self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)) < DTYPE_CMP_TOL)
            if self.anot0_d0 == C_CACHE_LEVEL:
                for k in self.coeffs.anot0_d0:
                    assert(abs(self.f[k] - self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)) < DTYPE_CMP_TOL)
            if self.anot0_d1 == C_CACHE_LEVEL:
                for k in self.coeffs.anot0_d1:
                    assert(abs(self.f[k] - self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)) < DTYPE_CMP_TOL)
            if self.a0_dnot01 == C_CACHE_LEVEL:
                for k in self.coeffs.a0_dnot01:
                    assert(abs(self.f[k] - self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)) < DTYPE_CMP_TOL)
            if self.a0_d1 == C_CACHE_LEVEL:
                for k in self.coeffs.a0_d1:
                    assert(abs(self.f[k] - self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)) < DTYPE_CMP_TOL)
            if self.a0_d0 == C_CACHE_LEVEL:
                for k in self.coeffs.a0_d0:
                    assert(abs(self.f[k] - self.get_f_c(&self.X[0, 0]+k*self.n_features_total, k)) < DTYPE_CMP_TOL)

            if self.anot0_dnot01 == C_STAR_CACHE_LEVEL:
                for k in self.coeffs.anot0_dnot01:
                    assert(abs(self.f_star[k] - self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)) < DTYPE_CMP_TOL)
            if self.anot0_d0 == C_STAR_CACHE_LEVEL:
                for k in self.coeffs.anot0_d0:
                    assert(abs(self.f_star[k] - self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)) < DTYPE_CMP_TOL)
            if self.anot0_d1 == C_STAR_CACHE_LEVEL:
                for k in self.coeffs.anot0_d1:
                    assert(abs(self.f_star[k] - self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)) < DTYPE_CMP_TOL)
            if self.a0_dnot01 == C_STAR_CACHE_LEVEL:
                for k in self.coeffs.a0_dnot01:
                    assert(abs(self.f_star[k] - self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)) < DTYPE_CMP_TOL)
            if self.a0_d1 == C_STAR_CACHE_LEVEL:
                for k in self.coeffs.a0_d1:
                    assert(abs(self.f_star[k] - self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)) < DTYPE_CMP_TOL)
            if self.a0_d0 == C_STAR_CACHE_LEVEL:
                for k in self.coeffs.a0_d0:
                    assert(abs(self.f_star[k] - self.get_f_star_c(&self.X[0, 0]+k*self.n_features_total+self.n_features, k)) < DTYPE_CMP_TOL)
            

    cdef two_dimensional_step_c(self, Py_ssize_t i, Py_ssize_t j):
        cdef DTYPE_t* x_i = &self.X[0, 0]+i*self.n_features_total
        cdef DTYPE_t* x_j = &self.X[0, 0]+j*self.n_features_total
        cdef DTYPE_t Kij = self.kmKij.get_c(x_i, x_j, self.n_features, i, j)

        x_i = &self.X[0, 0]+i*self.n_features_total+self.n_features
        x_j = &self.X[0, 0]+j*self.n_features_total+self.n_features
        cdef DTYPE_t Kij_star = self.kmKij_star.get_c(x_i, x_j, self.n_features_star, i, j)

        cdef DTYPE_t Kij_sum = Kij + Kij_star

        cdef DTYPE_t Delta_f      = self.f[i] - self.f[j]
        cdef DTYPE_t Delta_f_star = self.f_star[i] - self.f_star[j]
        cdef DTYPE_t Delta_f_sum = Delta_f + Delta_f_star

        cdef DTYPE_t t  = - Delta_f / Kij
        cdef DTYPE_t s0 =  Delta_f_star / Kij_star
        cdef DTYPE_t s  =  s0 + t
        cdef DTYPE_t t_s  =  t - s

        cdef DTYPE_t Delta_fx2 = 2*Delta_f
        cdef DTYPE_t Delta_f_starx2 = 2*Delta_f_star

        cdef DTYPE_t ai = self.coeffs.a[i]
        cdef DTYPE_t aj = self.coeffs.a[j]
        cdef DTYPE_t di = self.coeffs.d[i]
        cdef DTYPE_t dj = self.coeffs.d[j]

        cdef bint l_tmin_smin = True
        cdef bint l_tmin_smax = True
        cdef bint l_tmax_smin = True
        cdef bint l_tmax_smax = True

        cdef DTYPE_t Phi_opt = NP_NAN, Phi
        cdef DTYPE_t t_opt = NP_NAN, s_opt = NP_NAN
        cdef bint l_tmin=False, l_tmax=False, l_smin=False, l_smax=False

        cdef bint smin_i = -di > dj-1
        cdef DTYPE_t smin = -di if smin_i else dj-1

        cdef bint smax_i = 1-di < dj
        cdef DTYPE_t smax = 1-di if smax_i else dj

        IF DEBUG:
            assert(Delta_f_star > self.tau)
            assert(Delta_f_sum > self.tau)

        if t > -ai and t < aj and s > smin and s < smax: # optimum inside rectangle
            IF DEBUG:
                if self.logging:
                    logging.debug('optimum is inside rectangle')
            t_opt = t
            s_opt = s
            self.coeffs.set_anot0_c(i, ai + t_opt)
            self.coeffs.set_anot0_c(j, aj - t_opt)
            self.coeffs.set_dnot01_c(i, di + s_opt)
            self.coeffs.set_dnot01_c(j, dj - s_opt)
        else: # optimum on border
            IF DEBUG:
                if self.logging:
                    logging.debug('optimum is on the rectangle sides')
            # optimum on one of 4 edges
            t = -ai
            s = s0 + t
            if smin < s and s < smax:
                l_tmin_smin = False
                l_tmin_smax = False
                Phi_opt = t*(t*Kij + Delta_fx2) + s0*(s0*Kij_star - Delta_f_starx2)
                t_opt = t
                s_opt = s
                l_tmin = True
                l_tmax = False
                l_smin = False
                l_smax = False

            s = smin
            t = - (Delta_f_sum - s * Kij_star) / Kij_sum
            if -ai < t and t < aj:
                l_tmin_smin = False
                l_tmax_smin = False
                t_s = t - s 
                Phi = t*(t*Kij + Delta_fx2) + t_s*(t_s*Kij_star + Delta_f_starx2)
                if np.isnan(Phi_opt) or Phi < Phi_opt:
                    Phi_opt = Phi
                    t_opt = t
                    s_opt = s
                    l_tmin = False
                    l_tmax = False
                    l_smin = True
                    l_smax = False

            t = aj
            s = s0 + t
            if smin < s and s < smax:
                l_tmax_smin = False
                l_tmax_smax = False
                Phi = t*(t*Kij + Delta_fx2) + s0*(s0*Kij_star - Delta_f_starx2)
                if np.isnan(Phi_opt) or Phi < Phi_opt:
                    Phi_opt = Phi
                    t_opt = t
                    s_opt = s
                    l_tmin = False
                    l_tmax = True
                    l_smin = False
                    l_smax = False

            s = smax
            t = - (Delta_f_sum - s * Kij_star) / Kij_sum
            if -ai < t and t < aj:
                l_tmin_smax = False
                l_tmax_smax = False
                t_s = t - s 
                Phi = t*(t*Kij + Delta_fx2) + t_s*(t_s*Kij_star + Delta_f_starx2)
                if np.isnan(Phi_opt) or Phi < Phi_opt:
                    Phi_opt = Phi
                    t_opt = t
                    s_opt = s
                    l_tmin = False
                    l_tmax = False
                    l_smin = False
                    l_smax = True

            # optimum among 4 vertices
            if l_tmin_smin:
                t = -ai
                s = smin
                t_s = t - s
                Phi = t*(t*Kij + Delta_fx2) + t_s*(t_s*Kij_star + Delta_f_starx2)
                if np.isnan(Phi_opt) or Phi < Phi_opt:
                    Phi_opt = Phi
                    t_opt = t
                    s_opt = s
                    l_tmin = True
                    l_tmax = False
                    l_smin = True
                    l_smax = False

            if l_tmin_smax:
                t = -ai
                s = smax
                t_s = t - s
                Phi = t*(t*Kij + Delta_fx2) + t_s*(t_s*Kij_star + Delta_f_starx2)
                if np.isnan(Phi_opt) or Phi < Phi_opt:
                    Phi_opt = Phi
                    t_opt = t
                    s_opt = s
                    l_tmin = True
                    l_tmax = False
                    l_smin = False
                    l_smax = True

            if l_tmax_smin:
                t = aj
                s = smin
                t_s = t - s
                Phi = t*(t*Kij + Delta_fx2) + t_s*(t_s*Kij_star + Delta_f_starx2)
                if np.isnan(Phi_opt) or Phi < Phi_opt:
                    Phi_opt = Phi
                    t_opt = t
                    s_opt = s
                    l_tmin = False
                    l_tmax = True
                    l_smin = True
                    l_smax = False

            if l_tmax_smax:
                t = aj
                s = smax
                t_s = t - s
                Phi = t*(t*Kij + Delta_fx2) + t_s*(t_s*Kij_star + Delta_f_starx2)
                if np.isnan(Phi_opt) or Phi < Phi_opt:
                    Phi_opt = Phi
                    t_opt = t
                    s_opt = s
                    l_tmin = False
                    l_tmax = True
                    l_smin = False
                    l_smax = True

            IF DEBUG:
                assert(l_tmin or l_tmax or l_smin or l_smax)
                assert(not np.isnan(t_opt) and not np.isnan(s_opt))
                assert(t_opt*(t_opt*Kij + Delta_fx2) + (t_opt-s_opt)*((t_opt-s_opt)*Kij_star + Delta_f_starx2) <= 0.0)

            # optimum is found, correct a, d coeffs
            if l_tmin:
                self.coeffs.set_a0_c(i)
            else:
                self.coeffs.set_anot0_c(i, ai+t_opt)

            if l_tmax:
                self.coeffs.set_a0_c(j)
            else:
                self.coeffs.set_anot0_c(j, aj-t_opt)

            if l_smin:
                if smin_i:   #s_opt = -di > dj-1
                    self.coeffs.set_d0_c(i)
                    self.coeffs.set_dnot01_c(j, dj - s_opt)
                elif smax_i: #s_opt = dj-1 > -di  ( 1-di < dj )
                    self.coeffs.set_dnot01_c(i, di + s_opt)
                    self.coeffs.set_d1_c(j)
                else:        #s_opt = -di == dj-1
                    self.coeffs.set_d0_c(i)
                    self.coeffs.set_d1_c(j)
            elif l_smax: 
                if smin_i:   #s_opt = dj < 1-di   ( -di > dj-1 )
                    self.coeffs.set_dnot01_c(i, di + s_opt)
                    self.coeffs.set_d0_c(j)
                elif smax_i: #s_opt = 1-di < dj    
                    self.coeffs.set_d1_c(i)
                    self.coeffs.set_dnot01_c(j, dj - s_opt)
                else:        #s_opt = 1-di == dj
                    self.coeffs.set_d1_c(i)
                    self.coeffs.set_d0_c(j)
            else: # smin < s < smax
                self.coeffs.set_dnot01_c(i, di + s_opt)
                self.coeffs.set_dnot01_c(j, dj - s_opt)

        IF DEBUG:
            if self.logging:
                logging.debug('sum(alpha_k): '+str(sum([v for v in self.coeffs.a])))
                logging.debug('sum(delta_k): '+str(sum([v for v in self.coeffs.d])))

        self.f_update_c(i, j, t_opt)
        self.f_star_update_c(i, j, s_opt-t_opt)
        self.last_f_recalculate_level = NONE_ELEM_LEVEL

        IF DEBUG:
            self.check_ff_update_c()

    cdef alpha_2D_step_c(self, Py_ssize_t i, Py_ssize_t j):
        if self.alg == ALG_BEST_STEP_2D and \
           not self.coeffs.dis1[i] and not self.coeffs.dis0[j] and self.f_star[i] - self.f_star[j] > self.tau:
            if self.logging:
                logging.info('two-dimensional step')
            self.two_dimensional_step_c(i, j)
        else:
            self.alpha_step_c(i, j)

    cdef delta_2D_step_c(self, Py_ssize_t m, Py_ssize_t n):
        cdef int m_level = self.get_ff_cache_level(m)
        cdef int n_level = self.get_ff_cache_level(n)
        if self.alg == ALG_BEST_STEP_2D and \
            ((m_level == C_CACHE_LEVEL and n_level == C_CACHE_LEVEL) or \
             (m_level >= self.last_f_recalculate_level and n_level >= self.last_f_recalculate_level and self.last_f_recalculate_level != NONE_ELEM_LEVEL)) and \
            not self.coeffs.ais0[m] and self.f[m] + self.f_star[m] - self.f[n] - self.f_star[n] > self.tau:
            if self.logging:
                logging.info('two-dimensional step')
            if DEBUG:
                if self.logging:
                    if m_level == C_CACHE_LEVEL and n_level == C_CACHE_LEVEL:
                        logging.info('mn-pair is from C')
            self.two_dimensional_step_c(m, n)
        else:
            self.delta_step_c(m, n)

    cdef level_size_c(self, int ff_cache_level):
        cdef int s = 0
        if self.anot0_dnot01 == ff_cache_level:
            s += self.coeffs.anot0_dnot01.size()
        if self.anot0_d0 == ff_cache_level:
            s += self.coeffs.anot0_d0.size()
        if self.anot0_d1 == ff_cache_level:
            s += self.coeffs.anot0_d1.size()
        if self.a0_dnot01 == ff_cache_level:
            s += self.coeffs.a0_dnot01.size()
        if self.a0_d1 == ff_cache_level:
            s += self.coeffs.a0_d1.size()
        if self.a0_d0 == ff_cache_level:
            s += self.coeffs.a0_d0.size()            
        return s

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef fit_c(self, cnp.ndarray[DTYPE_t, ndim=2] X):
        IF DEBUG:
            if self.logging:
                logging.debug('fit_c')

        self.initialize_c(X)
        cdef int C_size, C_star_size, All_elem_size

        cdef int i = 0
        while True:
            C_size = self.level_size_c(C_CACHE_LEVEL)
            C_star_size = C_size + self.level_size_c(C_STAR_CACHE_LEVEL)
            All_elem_size = C_star_size + self.level_size_c(ALL_ELEM_LEVEL)

            if self.logging:
                logging.info('iteration: '+str(i))
                logging.info('alpha > 0: '+str(self.coeffs.a_not0_size())+', 0 < delta < 1: '+str(self.coeffs.d_not01_size())+', delta > 0: '+str(self.coeffs.d_not0_size())+', alpha=0, delta=0: '+str(self.coeffs.a0_d0.size())+', alpha > 0 and 0 < delta < 1: '+str(self.coeffs.anot0_dnot01.size()))
                logging.info('C size: '+str(C_size)+', C^* size: '+str(C_star_size)+', Total size: '+str(All_elem_size))

            if not self.make_best_step(alpha_pair_level = C_CACHE_LEVEL if C_size > 0 else NONE_ELEM_LEVEL, 
                                       delta_pair_level = C_STAR_CACHE_LEVEL if C_star_size > 0 else NONE_ELEM_LEVEL):
                if self.logging:
                    logging.info('neither alpha- nor delta-pair is violating')

                if not self.make_best_step(alpha_pair_level = C_STAR_CACHE_LEVEL if C_star_size > C_size else NONE_ELEM_LEVEL, 
                                           f_recalculate_level = C_STAR_CACHE_LEVEL if C_star_size > C_size else NONE_ELEM_LEVEL):
                    if self.logging:
                        logging.info('alpha-pair is not violating')

                    if not self.make_best_step(ALL_ELEM_LEVEL if All_elem_size > C_star_size else NONE_ELEM_LEVEL, 
                                               ALL_ELEM_LEVEL if All_elem_size > C_star_size else NONE_ELEM_LEVEL, 
                                               f_recalculate_level = ALL_ELEM_LEVEL if All_elem_size > C_star_size else NONE_ELEM_LEVEL, 
                                               f_star_recalculate_level = ALL_ELEM_LEVEL if All_elem_size > C_star_size else NONE_ELEM_LEVEL):
                        if self.logging:
                            logging.info('stop')
                        break
            i = i + 1
            if i == self.max_iter:
                break
        self.calc_b_star_c()
        self.calc_rho_c()

    def make_best_step(self, 
                       int alpha_pair_level=NONE_ELEM_LEVEL, 
                       int delta_pair_level=NONE_ELEM_LEVEL, 
                       int f_recalculate_level = NONE_ELEM_LEVEL, 
                       int f_star_recalculate_level = NONE_ELEM_LEVEL):
        if self.alg == ALG_DELTA_STEP_PREF:
            return self.make_best_step_0(alpha_pair_level, delta_pair_level, f_recalculate_level, f_star_recalculate_level)
        else:
            return self.make_best_step_12(alpha_pair_level, delta_pair_level, f_recalculate_level, f_star_recalculate_level)

    def make_best_step_0(self, 
                         int alpha_pair_level=NONE_ELEM_LEVEL, 
                         int delta_pair_level=NONE_ELEM_LEVEL, 
                         int f_recalculate_level=NONE_ELEM_LEVEL, 
                         int f_star_recalculate_level=NONE_ELEM_LEVEL):
        cdef PAIR_IJ ij
        cdef PAIR_IJ mn
        cdef DTYPE_t D_D_star = 0
        cdef DTYPE_t D_star = 0
        level = ['all elements', 'C^*', 'C']

        self.recalculate_f_star(f_star_recalculate_level)
        self.recalculate_f(f_recalculate_level)
        if delta_pair_level != NONE_ELEM_LEVEL:
            mn = self.delta_pair_c(delta_pair_level)
            if mn.first >= 0 and mn.second >= 0:
                D_star = self.f_star[mn.first] - self.f_star[mn.second]
            if self.logging:
                logging.info('delta-pair over '+level[delta_pair_level]+': '+str(mn)+', \Delta f^*_mn: '+str(D_star))

        if D_star > self.tau:
            if self.logging:
                logging.info('delta-pair is violating')
            self.delta_step_c(mn.first, mn.second)
        else:
            if alpha_pair_level != NONE_ELEM_LEVEL:
                ij = self.alpha_pair_c(alpha_pair_level)
                if ij.first >= 0 and ij.second >= 0:
                    D_D_star = self.f[ij.first] - self.f[ij.second] + self.f_star[ij.first] - self.f_star[ij.second]
                if self.logging:
                    logging.info('alpha-pair over '+level[alpha_pair_level]+': '+str(ij)+', \Delta f_ij + \Delta f^*_mn: '+str(D_D_star))

            if D_D_star > self.tau:
                if self.logging:
                    logging.info('alpha-pair is violating')
                self.alpha_step_c(ij.first, ij.second)
            else:
                return False
        return True

    def make_best_step_12(self, 
                          int alpha_pair_level=NONE_ELEM_LEVEL, 
                          int delta_pair_level=NONE_ELEM_LEVEL, 
                          int f_recalculate_level = NONE_ELEM_LEVEL, 
                          int f_star_recalculate_level = NONE_ELEM_LEVEL):
        cdef PAIR_IJ ij
        cdef PAIR_IJ mn
        cdef DTYPE_t D_D_star = 0
        cdef DTYPE_t D_star = 0
        level = ['all elements', 'C^*', 'C']

        self.recalculate_f(f_recalculate_level)
        self.recalculate_f_star(f_star_recalculate_level)

        if alpha_pair_level != NONE_ELEM_LEVEL:
            ij = self.alpha_pair_c(alpha_pair_level)
            if ij.first >= 0 and ij.second >= 0:
                D_D_star = self.f[ij.first] - self.f[ij.second] + self.f_star[ij.first] - self.f_star[ij.second]
            if self.logging:
                logging.info('alpha-pair over '+level[alpha_pair_level]+': '+str(ij)+', \Delta f_ij + \Delta f^*_mn: '+str(D_D_star))

        if delta_pair_level != NONE_ELEM_LEVEL:
            mn = self.delta_pair_c(delta_pair_level)
            if mn.first >= 0 and mn.second >= 0:
                D_star = self.f_star[mn.first] - self.f_star[mn.second]
            if self.logging:
                logging.info('delta-pair over '+level[delta_pair_level]+': '+str(mn)+', \Delta f^*_mn: '+str(D_star))

        if D_D_star > self.tau and D_star > self.tau:
            if D_D_star > D_star:
                if self.logging:
                    logging.info('alpha-pair is the worst violating')
                self.alpha_2D_step_c(ij.first, ij.second)
            else:
                if self.logging:
                    logging.info('delta-pair is the worst violating')
                self.delta_2D_step_c(mn.first, mn.second)
        elif D_D_star > self.tau:
            if self.logging:
                logging.info('only alpha-pair is violating')
            self.alpha_2D_step_c(ij.first, ij.second)
        elif D_star > self.tau:
            if self.logging:
                logging.info('only delta-pair is violating')
            self.delta_2D_step_c(mn.first, mn.second)
        else:
            return False
        return True

    cdef calc_b_star_c(self):
        s = []
        cdef Py_ssize_t i
        cdef PAIR_IJ mn
        if self.coeffs.d_not01_size() > 0:
            for i in self.coeffs.anot0_dnot01:
                s = s + [-self.f_star[i]]
            for i in self.coeffs.a0_dnot01:
                s = s + [-self.f_star[i]]
            self.b_star = np.mean(s)
        else:
            if self.logging:
                logging.warning('no 0 < \delta < 1 to average over -f* for b^* calculation!')
            mn = self.delta_pair_c(ALL_ELEM_LEVEL)
            self.b_star = - (self.f_star[mn.first] + self.f_star[mn.second]) / 2 # f_star[m] and f_star[n] are calculated 
        if self.logging:
            logging.info('b^*: '+str(self.b_star))
        IF DEBUG:
            if self.logging:
                logging.debug('std[b_k^*]: '+str(np.std(s)))


    cdef calc_rho_c(self):
        s = []
        cdef Py_ssize_t i
        for i in self.coeffs.anot0_dnot01:
            s = s + [self.f[i]+self.f_star[i]+self.b_star]
        for i in self.coeffs.anot0_d0:
            s = s + [self.f[i]+self.f_star[i]+self.b_star]
        for i in self.coeffs.anot0_d1:
            s = s + [self.f[i]+self.f_star[i]+self.b_star]
        self.rho = np.mean(s)
        if self.logging:
            logging.info('rho: '+str(self.rho))
        IF DEBUG:
            if self.logging:
                logging.debug('std[rho_k]: '+str(np.std(s)))


    def correcting_function(self, X_star):
        IF DEBUG:
            assert(X_star.shape[1] >= self.n_features_star)
        cdef Py_ssize_t i, n_samples = X_star.shape[0]
        cdef DTYPE_t[::1] x_c
        f_star = np.empty(n_samples, DTYPE)
        for i, x in enumerate(X_star):
            x_c = check_array(x[-self.n_features_star:])
            f_star[i] = self.get_f_star_c(&x_c[0]) + self.b_star
        return f_star

    def decision_function(self, X):
        IF DEBUG:
            assert(X.shape[1] >= self.n_features)
        cdef Py_ssize_t i, n_samples = X.shape[0]
        cdef DTYPE_t[::1] x_c
        f = np.empty(n_samples, DTYPE)
        for i, x in enumerate(X):
            x_c = check_array(x)
            f[i] = self.get_f_c(&x_c[0]) - self.rho
        return f

    def fit(self, X, y=None):
        return self.fit_c(X)

    def predict(self, X):
        return np.array([1 if f > 0 else -1 for f in self.decision_function(X)] )


class OCSVM_PLUS(BaseEstimator):
    def __init__(
            self,
            n_features,
            kernel='rbf',
            kernel_gamma='scale',
            kernel_star='rbf',
            kernel_star_gamma='scale',
            nu=0.5,
            gamma='auto',
            alg='best_step_2d',       # ('best_step', 'delta_pair')
            tau=0.001,
            ff_caches='not_bound',    # ('all', 'not_zero')
            kernel_cache_size=0, 
            distance_cache_size=0,
            max_iter=-1,
            random_seed=None,
            logging_file_name=None):
        self.n_features = n_features
        self.n_features_star = None
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma
        self.kernel_star = kernel_star
        self.kernel_star_gamma = kernel_star_gamma
        self.nu = nu
        self.gamma = gamma
        self.alg = alg
        self.tau = tau
        self.ff_caches = ff_caches
        self.kernel_cache_size = kernel_cache_size
        self.distance_cache_size = distance_cache_size
        self.random_seed = random_seed
        self.logging_file_name = logging_file_name
        self.max_iter = max_iter
        self.is_fitted_ = False
        self.model_ = None

        if not (self.n_features > 0 and self.n_features == int(self.n_features)):
            raise ValueError("n_features must be positive integer!")

        if not (self.nu > 0 and self.nu < 1):
            raise ValueError("must be 0<\nu <1!")

        if not self.tau > 0:
            raise ValueError("bad input for tau!")

        if not self.kernel_cache_size >= 0:
            raise ValueError("bad input for kernel_cache_size!")

        if not self.distance_cache_size >= 0:
            raise ValueError("bad input for distance_cache_size!")

        if not self.max_iter >= -1:
            raise ValueError("bad input for max_iter!")

        if not (self.gamma == 'auto' or self.gamma > 0):
            raise ValueError("bad input for gamma!")

        if not (self.kernel_gamma == 'scale' or self.kernel_gamma == 'auto' or self.kernel_gamma > 0):
            raise ValueError("bad input for kernel_gamma!")

        if not (self.kernel_star_gamma == 'scale' or self.kernel_star_gamma == 'auto' or self.kernel_star_gamma > 0):
            raise ValueError("bad input for kernel_star_gamma!")

        if not (self.kernel == 'rbf' or self.kernel == 'linear' or isinstance(self.kernel, kernel)):
            raise ValueError("bad input for kernel!")

        if not (self.kernel_star == 'rbf' or self.kernel_star == 'linear' or isinstance(self.kernel_star, kernel)):
            raise ValueError("bad input for kernel_star!")

        if not( (isinstance(self.ff_caches, dict) and 
                 set(self.ff_caches.keys()) == {'anot0_dnot01', 'anot0_d0', 'anot0_d1', 'a0_dnot01', 'a0_d1', 'a0_d0'} and
                 set(self.ff_caches.values()).issubsetof({0, 1, 2})) or 
                self.ff_caches in ['all', 'not_bound', 'not_zero']):
            raise ValueError("bad input for ff_caches!")


    def fit(self, X, y=None):
        X0 = check_array(X, dtype=DTYPE, order='C')
        if not self.n_features < X0.shape[1]:
            raise ValueError("number of features in X must be greater than n_features!")
        n_samples = X0.shape[0]
        self.n_features_star = X0.shape[1] - self.n_features
        X = X0[:, :self.n_features]
        X_star = X0[:, -self.n_features_star:]
        
        if self.gamma == 'auto':
            gamma = self.nu * n_samples
        else:
            gamma = self.gamma

        if self.kernel_gamma == 'scale':
            kernel_gamma = 1.0 / (self.n_features * X.var())
        elif self.kernel_gamma == 'auto':
            kernel_gamma = 1.0 / self.n_features
        else:
            kernel_gamma = self.kernel_gamma

        if self.kernel_star_gamma == 'scale':
            kernel_star_gamma = 1.0 / (self.n_features_star * X_star.var())
        elif self.kernel_star_gamma == 'auto':
            kernel_star_gamma = 1.0 / self.n_features_star
        else:
            kernel_star_gamma = self.kernel_star_gamma

        if self.kernel == 'rbf':
            K = kernel_rbf(kernel_gamma)
        elif self.kernel == 'linear':
            K = kernel_linear()
        else:
            K = self.kernel

        if self.kernel_star == 'rbf':
            K_star = kernel_rbf(kernel_star_gamma)
        elif self.kernel_star == 'linear':
            K_star = kernel_linear()
        else:
            K_star = self.kernel_star

        if self.ff_caches == 'not_bound':
            ff_caches = {'anot0_dnot01': 2, 'anot0_d0': 2, 'anot0_d1': 2, 'a0_dnot01': 2, 'a0_d1': 1, 'a0_d0': 1}
        elif self.ff_caches == 'all':
            ff_caches = {'anot0_dnot01': 2, 'anot0_d0': 2, 'anot0_d1': 2, 'a0_dnot01': 2, 'a0_d1': 2, 'a0_d0': 2}
        elif self.ff_caches == 'not_zero':
            ff_caches = {'anot0_dnot01': 2, 'anot0_d0': 2, 'anot0_d1': 2, 'a0_dnot01': 2, 'a0_d1': 2, 'a0_d0': 0}
        else:
            ff_caches = self.ff_caches

        if self.alg == 'best_step_2d':
            alg = ALG_BEST_STEP_2D
        elif self.alg == 'best_step':
            alg = ALG_BEST_STEP
        elif self.alg == 'delta_pair':
            alg = ALG_DELTA_STEP_PREF
        else:
            raise ValueError("bad input for alg!")

        self.model_ = OCSVM_PLUS_C(n_features=self.n_features,
                                   nu=self.nu,
                                   gamma=gamma,
                                   tau=self.tau,
                                   K=K,
                                   K_star=K_star,
                                   k_cache_size=self.kernel_cache_size,
                                   kii_ij_jj_cache_size=self.distance_cache_size,
                                   ff_caches=ff_caches,
                                   alg=alg,
                                   max_iter=self.max_iter,
                                   random_seed=self.random_seed,
                                   logging_file_name=self.logging_file_name)

        self.model_.fit(X0, y)
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        check_is_fitted(self, 'is_fitted_')
        if X.shape[1] < self.n_features:
            raise ValueError("no enough original features!")
        self.model_.decision_function(X)

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        if X.shape[1] < self.n_features:
            raise ValueError("no enough original features!")
        self.model_.predict(X)

    def correcting_function(self, X_star):
        check_is_fitted(self, 'is_fitted_')
        if X_star.shape[1] < self.n_features_star:
            raise ValueError("not enough privileged features!")
        self.model_.decision_function(X_star)

    @property
    def alphas_(self):
        return np.array(self.model_.coeffs.a, dtype=DTYPE)

    @property
    def deltas_(self):
        return np.array(self.model_.coeffs.d, dtype=DTYPE)

    @property
    def rho_(self):
        return self.model_.rho

    @property
    def b_star_(self):
        return self.model_.b_star

    @property
    def fit_status_(self):
        return self.alpha_support_.size > 0 and self.delta_support_.size > 0

    @property
    def alpha_support_(self):
        coeffs = self.model_.coeffs
        return np.array(sorted(list(coeffs.anot0_dnot01) + list(coeffs.anot0_d0) + list(coeffs.anot0_d1)))

    @property
    def delta_support_(self):
        coeffs = self.model_.coeffs
        return np.array(sorted(list(coeffs.anot0_dnot01) + list(coeffs.a0_dnot01)))

if __name__ == "__main__":
    pass