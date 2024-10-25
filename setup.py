from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension
ext_modules = [
    Extension(
        "chi2sim.chi_square_mc",
        sources=[
            "chi2sim/chi_square_mc.pyx",
            "chi2sim/src/chi_square_mc.c"
        ],
        include_dirs=[
            np.get_include(),
            "chi2sim/src"
        ],
        depends=["chi2sim/src/chi_square_mc.h"]
    )
]

# Setup configuration
setup(
    packages=["chi2sim"],
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ),
)
