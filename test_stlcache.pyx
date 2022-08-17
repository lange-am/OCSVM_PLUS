# STLCACHE tests
#
# Copyright (C) 2021 Andrey M. Lange, 
#
# cython: language_level=3
# distutils: language = c++

from stlcache cimport cache
from stlcache cimport policy_lru
from stlcache cimport policy_lfu

cdef cache[int, double, policy_lru] *clru = new cache[int, double, policy_lru](10)
cdef cache[int, double, policy_lfu] *clfu = new cache[int, double, policy_lfu](10)        

for i in range(50):
    exist, _ = clru.recorded_find(0)
    if not exist:
        clru.insert(0, 0)
    exist, _ = clfu.recorded_find(0)
    if not exist:
        clfu.insert(0, 0)
        
for i in range(1, 10):
    exist, _ = clru.recorded_find(i)
    if not exist:
        clru.insert(i, i*0.1)
    exist, _ = clfu.recorded_find(i)
    if not exist:
        clfu.insert(i, i*0.1)
        
exist, _ = clru.recorded_find(10)
if not exist:
    clru.insert(10, 10*0.1)
exist, _ = clfu.recorded_find(10)
if not exist:
    clfu.insert(10, 10*0.1)
        
lru_incache, lfu_incache = [], []
for i in range(11):
    exist, _ = clru.recorded_find(i)
    lru_incache.append(exist)
    exist, _ = clfu.recorded_find(i)
    lfu_incache.append(exist)

# As expected, different policies get different results
TEST_LRU_LFU_RES = lru_incache == [False]+[True]*10 and lfu_incache == [True, False]+[True]*9

del clru
del clfu

if __name__ == "__main__":
    pass