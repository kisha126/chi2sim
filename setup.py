import sys
from setuptools import setup, Extension, find_packages  # Added find_packages
from Cython.Build import cythonize
import numpy as np

# Define the extension
ext_modules = [
    Extension(
        "chi2sim.chi2_cont_sim",
        sources=[
            "chi2sim/chi2_cont_sim.pyx",
            "chi2sim/src/chi_square_mc.c"
        ],
        include_dirs=[
            np.get_include(),
            "chi2sim/src"
        ],
        depends=["chi2sim/src/chi_square_mc.h"],
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O2"]
    )
]

# Setup configuration
setup(
    name="chi2sim",
    version="1.0.2.3",  # Increment version
    packages=find_packages(),  # This will find chi2sim and chi2sim.src
    package_data={
        'chi2sim': ['src/*.h', 'src/*.c', '*.pyx'],  # Include C and header files
    },
    include_package_data=True,  # This ensures package_data is included
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ),
    install_requires=[
        "numpy>=1.19.2",
    ],
)
