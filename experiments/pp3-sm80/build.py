"""Compile locally with the selected runtime's PyTorch and matching CUDA toolkit."""
import os
from pathlib import Path
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parent
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
os.environ.setdefault('MAX_JOBS','2')
(root/'build').mkdir(exist_ok=True)
load(name='q6_speed_cuda',sources=[str(root/'q6_moe_speed.cu')],
     extra_include_paths=[str(root)],extra_cflags=['-O3'],
     extra_cuda_cflags=['-O3','--use_fast_math','-std=c++17'],
     build_directory=str(root/'build'),is_python_module=False,verbose=True)
