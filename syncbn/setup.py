
import io
import os
import glob
import subprocess

from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

cwd = os.path.dirname(os.path.abspath(__file__))




def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))

    gpu_extensions_dir = os.path.join(this_dir, "lib")

    source_cuda = glob.glob(os.path.join(gpu_extensions_dir, "*.cpp")) + \
        glob.glob(os.path.join(gpu_extensions_dir, "*.cu"))

    print('cuda: ', source_cuda)
    extra_compile_args = {"cxx": []}

    if CUDA_HOME is not None:
        define_macros = [("WITH_CUDA", None)]
        include_dirs += [gpu_extensions_dir]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        ext_modules = [
            CUDAExtension(
                "syncbn.lib",
                source_cuda,
                include_dirs=include_dirs,
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
            )
        ]

    return ext_modules

if __name__ == '__main__':

    setup(
        name="torch-syncbn",
        package_data={ 'torch-segmentation': [
            'lib/*.h',
            'lib/*.cpp',
            'lib/*.h',
            'lib/*.cpp',
            'lib/*.cu',
        ]},
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )