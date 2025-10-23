from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, torch

def cxx11_abi():
    try:
        return int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception:
        return 1

this_dir = os.path.dirname(os.path.abspath(__file__))

extra_cflags = [
    "-O3", "-std=c++17",
    f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi()}",
]
extra_nvcc = [
    "-O3", "--use_fast_math", "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    # 建议用环境变量 TORCH_CUDA_ARCH_LIST 控制架构（比如 H100 -> "9.0"）
    # "-gencode=arch=compute_90,code=sm_90",
]

setup(
    name='emd_ext',
    ext_modules=[
        CUDAExtension(
            name='emd_ext',   # 统一用 emd_ext，避免与 emd_cuda.py 冲突
            sources=[
                os.path.join('cuda', 'emd.cpp'),
                os.path.join('cuda', 'emd_kernel.cu'),
            ],
            include_dirs=[os.path.join(this_dir, 'cuda')],  # 让编译器找到我们自带的 THC 兼容头
            extra_compile_args={'cxx': extra_cflags, 'nvcc': extra_nvcc},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
