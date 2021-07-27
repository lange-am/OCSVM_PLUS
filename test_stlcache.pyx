# STLCACHE tests
#
# Copyright (C) 2021 Andrey M. Lange, 
# Skoltech, https://crei.skoltech.ru/cdise
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at 
# http://www.boost.org/LICENSE_1_0.txt)
#
# Just import this file as a module: pyhon3 -c "import test_stlcache"
#
# cython: language_level=3
# distutils: language = c++


from stlcache cimport cache
from stlcache cimport policy_lru
from stlcache cimport policy_lfu

# As expected, different policies get different results:

#LRU:
cdef cache[int, double, policy_lru] *clru = new cache[int, double, policy_lru](10)

for i in range(50):
    exist, _ = clru.recorded_find(0)
    if not exist:
        clru.insert(0, 0)

for i in range(1, 10):
    exist, _ = clru.recorded_find(i)
    if not exist:
        clru.insert(i, i*0.1)

exist, _ = clru.recorded_find(10)
if not exist:
    clru.insert(10, 10*0.1)

print('As a sesult of LRU policy:')
for i in range(11):
    exist, _ = clru.recorded_find(i)
    print(i, '-th is in cache:', exist)

del clru

#LRU:
cdef cache[int, double, policy_lfu] *clfu = new cache[int, double, policy_lfu](10)

for i in range(50):
    exist, _ = clfu.recorded_find(0)
    if not exist:
        clfu.insert(0, 0)

for i in range(1, 10):
    exist, _ = clfu.recorded_find(i)
    if not exist:
        clfu.insert(i, i*0.1)

exist, _ = clfu.recorded_find(10)
if not exist:
    clfu.insert(10, 10*0.1)

print('As a sesult of LFU policy:')
for i in range(11):
    exist, _ = clfu.recorded_find(i)
    print(i, '-th is in cache:', exist)

del clfu

if __name__ == "__main__":
    pass