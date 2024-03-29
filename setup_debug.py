# OCSVM+: One-Class SVM with Privileged Information debug module build
#
# Copyright (C) 2021 Andrey M. Lange, 

# run this: python setup_debug.py build_ext --inplace

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy
import os

INCLUDE_FILE = 'debug'
with open(INCLUDE_FILE, 'w') as f:
    f.write('DEF DEBUG = True\n')

extensions = [Extension("test_stlcache", ["test_stlcache.pyx"])]
extensions += [Extension("ocsvm_plus_debug", ["ocsvm_plus.pyx"])]
setup(ext_modules=cythonize(extensions, annotate=True), include_dirs=[numpy.get_include()])

if os.path.exists(INCLUDE_FILE):
    os.remove(INCLUDE_FILE)
