"""Build the minimax C++ extension.

Usage:  python setup.py build_ext --inplace
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="minimax_cpp",
    ext_modules=[
        Pybind11Extension("minimax_cpp", ["minimax_bot.cpp"],
                          cxx_std=17,
                          extra_compile_args=["-O3", "-march=native", "-DNDEBUG"],
                          include_dirs=["."]),
    ],
    cmdclass={"build_ext": build_ext},
)
