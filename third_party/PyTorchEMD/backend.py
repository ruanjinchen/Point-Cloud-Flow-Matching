# third_party/PyTorchEMD/backend.py
import os, time
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_build_dir = os.path.join(_src_path, 'build_dynamic')
os.makedirs(_build_dir, exist_ok=True)

tic = time.time()

emd_cuda_dynamic = load(
    name='emd_ext',  # 与 setup.py 的扩展名保持一致
    sources=[
        os.path.join(_src_path, 'cuda', 'emd.cpp'),
        os.path.join(_src_path, 'cuda', 'emd_kernel.cu'),
    ],
    build_directory=_build_dir,
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=[
        '-O3', '--use_fast_math', '-std=c++17',
        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
        '--expt-relaxed-constexpr', '--expt-extended-lambda',
    ],
    extra_include_paths=[os.path.join(_src_path, 'cuda')],  # 命中我们本地的 THC 兼容头
    verbose=True,
)

print('load emd_ext time: {:.3f}s'.format(time.time() - tic))
__all__ = ['emd_cuda_dynamic']
