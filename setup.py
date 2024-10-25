from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension
ext_modules = [
    Extension(
        "chi2sim.chi2_cont_sim",
        sources=[
            "chi2sim/chi2_cont_sim.pyx",  # Updated filename
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
    name="chi2sim",
    version="1.0.0",
    packages=["chi2sim"],
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ),
    install_requires=[
        "numpy",
    ],
)
