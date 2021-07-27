# Cython interface to some from STLCACHE external library 
# 
# Copyright (C) 2021 Andrey M. Lange, 
# Skoltech, https://crei.skoltech.ru/cdise
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at 
# http://www.boost.org/LICENSE_1_0.txt)

# cython: language_level=3, boundscheck=False, wraparound=False


from libcpp.pair cimport pair

cdef extern from "stlcache/stlcache.hpp" namespace "stlcache":

    # cdef cppclass policy:             # abstract
    #     pass

    cdef cppclass policy_lfu:         # http://en.wikipedia.org/wiki/Least_frequently_used
        policy_lfu() except+

    cdef cppclass policy_lru:         # https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)
        policy_lru() except+

    cdef cppclass cache[Key, Data, Policy, Compare=*, Allocator=*]:
        # ctypedef Policy policy_type
        cache(Py_ssize_t) except +
        pair[bint, Data] recorded_find(const Key&) except +
        pair[bint, Data] unrecorded_find(const Key&) except +
        bint insert(Key, Data) except +
        size_t size() except +