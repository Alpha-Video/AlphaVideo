from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='nms_cuda',
    ext_modules=[
        CUDAExtension('tube_nms_cuda', [
            'src/nms_cuda.cpp',
            'src/nms_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
