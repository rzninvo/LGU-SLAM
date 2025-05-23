from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='defCorrSample',
    ext_modules=[
        CUDAExtension('defCorrSample',
            sources=[
                'droid.cpp', 
                'defCorrSample_kernel.cu',
                'lowMem_defSample.cu',
                'corrSample_kernel.cu',
                'gaussianAttn.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_86,code=sm_89',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
