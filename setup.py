# OCSVM+: One-Class SVM with Privileged Information module build
#
# Copyright (C) 2021 Andrey M. Lange, 
# Skoltech, https://crei.skoltech.ru/cdise
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at 
# http://www.boost.org/LICENSE_1_0.txt)

# run this: python setup_debug.py build_ext --inplace

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy
import os

INCLUDE_FILE = 'debug'
with open(INCLUDE_FILE, 'w') as f:
    f.write('DEF DEBUG = False')

extensions = [Extension("ocsvm_plus", ["ocsvm_plus.pyx"])]
setup(ext_modules=cythonize(extensions, annotate=True), include_dirs=[numpy.get_include()])

if os.path.exists(INCLUDE_FILE):
    os.remove(INCLUDE_FILE)
