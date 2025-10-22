# Point-Cloud-Flow-Matching  
This is a RGB Point Cloud Generation Model based on Flow Matching implemented using PyTorch. Tested on Windows11 CUDA 13.0 and Ubuntu22.04 CUDA 12.4.  
![A random sample eyeglass](./assets/pred_4.png)
## Installation
```sh
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
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