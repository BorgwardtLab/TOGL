from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='unionfind',
    version='1.0',
    description='Provides a fast disjoint set forest data structure.',
    author='Justin Eldridge',
    author_email='eldridge@cse.ohio-state.edu',
    ext_modules = cythonize("unionfind.pyx")
)
