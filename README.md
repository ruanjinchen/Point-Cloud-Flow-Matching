# Point-Cloud-Flow-Matching  
This is a RGB Point Cloud Generation Model based on Flow Matching implemented using PyTorch. Tested on Windows11 CUDA 13.0 and Ubuntu22.04 CUDA 12.4.  
![A random sample eyeglass](./assets/pred_4.png)
## Installation

```sh
pip install -r requirements.txt
```
**CHOOSE YOUR OWN CUDA VERSION!!! DO NOT RUN ALL THE COMMAND BELLOW!!!**
```sh
# PyTorch + cu124 Official Index
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision

# PyTorch + cu130 Official Index
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
  torch torchvision

#CUDA 12.4
conda install -y --override-channels --solver=libmamba \
  -c nvidia/label/cuda-12.4.1 -c defaults \
  cuda-toolkit=12.4.*

#CUDA 13.0
conda install -y --override-channels --solver=libmamba \
  -c nvidia/label/cuda-13.0.2 -c defaults \
  cuda-toolkit=13.0.*
```

## Compile
### PVCNN
```sh
cd third_party/pvcnn
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0"
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS=16

python - <<'PY'
import time
print("[Step] import modules.functional.backend 以触发JIT编译……")
t0=time.time()
from modules.functional import backend  # 这里会调用 torch.utils.cpp_extension.load(...)
print("[OK ] _pvcnn_backend 已加载，用时 %.1fs" % (time.time()-t0))
PY
```

### PyTorchEMD
```sh
cd third_party/PyTorchEMD

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0"
export MAX_JOBS=16

pip install -v .
```
Check
```sh
python - <<'PY'
import torch, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda)
# 确认能直接导入二进制扩展
m = importlib.import_module("emd_ext")
print("emd_ext ->", m.__file__)

from third_party.PyTorchEMD.emd import earth_mover_distance  # 这行会优先用 emd_ext
x = torch.rand(2, 1024, 3, device='cuda')
y = torch.rand(2, 1024, 3, device='cuda')
d = earth_mover_distance(x, y, transpose=False)
print("EMD OK:", d.shape, "mean:", float(d.mean()))
PY
```
### ChamferDistancePytorch
```sh
cd third_party/ChamferDistancePytorch/chamfer3D
pip install -v .

# First find torch/lib's abs path
python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
#Replace the path below
export LD_LIBRARY_PATH=/data/conda/envs/flow/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
```
Check (calculate two same pointclouds' CD using **chamfer_3D**, the result should be 0.)
```sh
python - <<'PY'
import torch, importlib
m = importlib.import_module('chamfer_3D')

B,N = 2,2048
x = torch.randn(B,N,3, device='cuda', dtype=torch.float32).contiguous()

d1 = torch.empty(B,N, device='cuda', dtype=torch.float32)
d2 = torch.empty(B,N, device='cuda', dtype=torch.float32)
i1 = torch.empty(B,N, device='cuda', dtype=torch.int32)
i2 = torch.empty(B,N, device='cuda', dtype=torch.int32)

ret = m.forward(x, x, d1, d2, i1, i2)
cd = (d1.mean() + d2.mean()).item()
print(f"ret={ret} | CD(same): {cd:.8f}")
PY
```

## Dataset

- Download PartNet Mobility Dataset from https://sapien.ucsd.edu/browse. Extract to dataset/partnet. 
- Create PartNet H5 Datasets. Using Scissors as an example.
```sh
cd dataset
python make_mobility_multijoint_colored.py \
  --index partnet_index.csv --dataset-dir partnet \
  --out-dir H5/Scissors \
  --filter-cats Scissors --joint-types revolute --joint-regex ".*" \
  --steps 50 --num-combos 50 --points 20000 --seed 0 \
  --ply-ascii --point-sampling random
```


## Train
My network has very low requirements for GPU memory. You can start a train with 8GB GPU memory using 10,000 points for training.  
✨✨✨**_Training with a point cloud of 20,000 points requires only 9GB of GPU memory. On one H100, an epoch takes only 25 seconds._**✨✨✨ 
```sh
export CUDA_VISIBLE_DEVICES=4
python train.py \
  --dataset_type partnet_h5 \
  --data_dir dataset/H5/Scissors \
  --batch_size 8 --epochs 3000 --save_every 100 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --tdcr_use_norm \
  --latent_dim 128 \
  --partnet_cond_policy mode \
  --lambda_pair 0.1 --lambda_var 1.0 --lambda_cov 0.01 --lambda_zreg 1e-4 \
  --lambda_adv 0.0 --lambda_color 1.0\
  --use_rgb_in_latent --pointflow_rgb \
  --color_prior uniform \
  --partnet_report_file_train runs/scissors_rgb/_train_report.json \
  --out_dir runs/scissors_rgb
```

## Sampling