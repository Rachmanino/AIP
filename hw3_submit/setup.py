from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, find_packages
import os

__version__ = '0.1.0'

dir = 'csrc'
src = ['{}/{}'.format(dir, src) for src in os.listdir(dir) if src.endswith('.cu') or src.endswith('.cpp')]

setup(
    name = 'backend',
    version=__version__,
    author='Tong WU',
    author_email='2200013212@stu.pku.edu.cn',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['torch', 'torchvision'],
    python_requires='>=3.10',
    license='MIT',
    ext_modules=[
        CUDAExtension(
            name = 'backend',
            sources = src,
            extra_compile_args = {
                'nvcc': ['--extended-lambda'] # to enable the use of lambda in CUDA
            }
        )
    ],
    include_dirs = ['/home/sp_test/anaconda/anaconda3/envs/aip/lib/python3.10/site-packages/torch/include', 
                    # path to torch in the conda env, which contains pybind11 folder.
                     '/home/sp_test/anaconda/anaconda3/envs/aip/include/python3.10',
                    # path to python3.10 include folder in the conda env
                     dir],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
)
