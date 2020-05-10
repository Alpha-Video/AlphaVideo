import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# VERSION = __import__("alphaVideo").__version__

VERSION = "0.1"
LICENSE = "MIT"

setup(
    name='alphavideo',
    version=VERSION,
    description=(
        'video understanding models proposed by MVIG in SJTU.'
    ),
    author='Bo Pang',
    author_email='',
    maintainer='',
    maintainer_email='',
    license=LICENSE,
    packages=find_packages(),
    url='http://mvig.sjtu.edu.cn',
    install_requires=[
        "easydict",
        "requests",
        "yacs"
    ],
    ext_modules=[
        CUDAExtension('alphavideo.utils.roi_align_3d_cuda', [
            'alphavideo/csrc/ROIAlign3d/ROIAlign3d_cuda.cpp',
            'alphavideo/csrc/ROIAlign3d/ROIAlign3d_kernel.cu',
        ]),
        CUDAExtension('alphavideo.utils.tube_nms_cuda', [
            'alphavideo/csrc/tube_nms_c/src/nms_cuda.cpp',
            'alphavideo/csrc/tube_nms_c/src/nms_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension}
)

# setup(
#     name='nms_cuda',
#     ext_modules=[
#         CUDAExtension('alphavideo.utils.tube_nms_cuda', [
#             'alphavideo/csrc/tube_nms_c/src/nms_cuda.cpp',
#             'alphavideo/csrc/tube_nms_c/src/nms_kernel.cu',
#         ]),
#     ],
#     cmdclass={'build_ext': BuildExtension})
