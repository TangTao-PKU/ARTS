# ARTS
<div align="center">

  <h1 align="center">ARTS: Semi-Analytical Regressor using Disentangled Skeletal Representations for Human Mesh Recovery from Videos (ACM MM 2024)</h1>
</div>
This is the offical Pytorch implementation of the paper: "ARTS: Semi-Analytical Regressor using Disentangled Skeletal Representations for Human Mesh Recovery from Videos"

## Statement
The project has undergone complete anonymization for reviewers to understand some implementation details that they may be interested in. However, due to the need for anonymization, dataset preparation and processing are unavailable yet and these will be fully open-sourced in subsequent rebuttals or after paper acceptance.

## Preparation

1. Install dependencies. This project is developed on Ubuntu 18.04 with NVIDIA 3090 GPUs. We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment.
```bash
# Create a conda environment.
conda create -n arts python=3.8
conda activate arts

# Install PyTorch >= 1.2 according to your GPU driver.
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install other dependencies.
sh requirements.sh
```
2. Prepare SMPL layer. 
- For the SMPL layer, We used [smplpytorch](https://github.com/gulvarol/smplpytorch). The repo is already included in `./smplpytorch` folder.
- Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPL](https://smpl.is.tue.mpg.de/downloads) (female & male) and [SMPL](http://smplify.is.tue.mpg.de/) (neutral) to `./smplpytorch/smplpytorch/native/models`.

## Implementation
### Data Preparation
The `./data` directory structure should follow the below hierarchy. All the processed annotation files will be available.
```
${Project}  
|-- data  
|   |-- base_data
|   |   |-- J_regressor_extra.npy
|   |   |-- mesh_downsampling.npz
|   |   |-- smpl_mean_params.npz
|   |   |-- smpl_mean_vertices.npy
|   |   |-- SMPL_NEUTRAL.pkl
|   |   |-- spin_model_checkpoint.pth.tar
|   |-- COCO  
|   |   |-- coco_data  
|   |   |-- __init__.py
|   |   |-- dataset.py
|   |   |-- J_regressor_coco.npy
|   |-- Human36M  
|   |   |-- h36m_data  
|   |   |-- __init__.py
|   |   |-- dataset.py 
|   |   |-- J_regressor_h36m_correct.npy
|   |   |-- noise_stats.py
|   |-- MPII  
|   |   |-- mpii_data  
|   |   |-- __init__.py
|   |   |-- dataset.py
|   |-- MPII3D
|   |   |-- mpii3d_data  
|   |   |-- __init__.py
|   |   |-- dataset.py
|   |-- PW3D 
|   |   |-- pw3d_data
|   |   |-- __init__.py
|   |   |-- dataset.py
|   |-- multiple_datasets.py
```

### Train
Stage 1 : Train the 3D pose estimation.
```bash
# Human3.6M
bash train_pose_h36m.sh

# 3DPW
bash train_pose_3dpw.sh
```

Stage 2: To train the all network for final mesh. Configs of the experiments can be found and edited in `./config` folder. Change `posenet_path` in `./config/train_mesh_*.yml` to the path of the pre-trained pose model.
```bash
# Human3.6M
bash train_mesh_h36m.sh

# 3DPW & MPII3D
bash train_mesh_3dpw.sh
```

### Test
To test on a pre-trained pose estimation model (Stage 1).
```bash
# Human3.6M
bash test_pose_h36m.sh

# 3DPW
bash test_pose_3dpw.sh
```

To test on a pre-trained mesh model (Stage 2).
```bash
# Human3.6M
bash test_mesh_h36m.sh

# 3DPW
bash test_mesh_3dpw.sh

# MPII3D
bash test_mesh_mpii3d.sh
```
Change the `weight_path` in the corresponding `./config/test_*.yml` to your model path.


