# PointSSM: State Space Model for Large-Scale LiDAR Point Cloud Semantic Segmentation

## Results
### Indoor semantic segmentation

|  Model   |   Benchmark   | Num GPUs | Val mIoU |                                       Tensorboard                                       |              Exp Record              |
|:--------:| :-----------: |:--------:|:--------:|:---------------------------------------------------------------------------------------:|:------------------------------------:|
| PointSSM |    ScanNet    |    2     |  78.1%   | [link](exp/scannet/semseg-default/events.out.tfevents.1728354922.cv-Z690-GAMING-X-DDR4) |  [link](exp/scannet/semseg-default)  |
|   PointSSM   |   ScanNet200  |    2     |  35.7%   |         [link](exp/scannet200/semseg-default/events.out.tfevents.1730197260.cv-Z690-GAMING-X-DDR4)         | [link](exp/scanne200/semseg-default) |
|   PointSSM   | S3DIS (Area5) |    2     |  72.8%   |         [link](exp/s3dis/semseg-default/events.out.tfevents.1728446362.cv-Z690-GAMING-X-DDR4)         |   [link](exp/s3dis/semseg-default)   |

### Outdoor semantic segmentation  
|    Model   |   Benchmark   | Num GPUs | Val mIoU |                                       Tensorboard                                        |             Exp Record              |
| :--------: |:-------------:|:--------:|:--------:|:----------------------------------------------------------------------------------------:|:-----------------------------------:|
|    PointSSM    |   nuScenes    |    2     |  80.7%   | [link](exp/nuscenes/semseg-default/events.out.tfevents.1728970636.cv-Z690-GAMING-X-DDR4) | [link](exp/nuscenes/semseg-default) |
|    PointSSM    | SemanticKITTI |    2     |    –     |                                            –                                             |                  –                  |
|    PointSSM    |     DALES     |    2     |  82.3%   |                                            -                                             |                  -                  |

## Data
* Scannet,S3DIS datasets can be downloaded following by (https://huggingface.co/Pointcept/datasets)
* DALES can be downloaded at (https://drive.google.com/file/d/1Ta5Hg7e1dyUCSaDRCfAt_sDyisBuoikw/view?usp=drive_link)


## Environment
Our database builds on [Pointcept](https://github.com/Pointcept/Pointcept>) codebase.
### Requirements
- Ubuntu: 18.04 and above.
- *CUDA: 11.8 and above.*
- PyTorch: 1.10.0 and above.
### Conda Environment

```bash
conda create -n pointssm python=3.8 -y
conda activate pointssm
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PPT (clip)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d

# Mamba-ssm
pip install mamba-ssm==1.0.1
```

## Training and testing
Please follow the Pointcept codebase.

## Acknowledgements
We thank the authors of [Point Transformer V3](https://github.com/Pointcept/Pointcept>). Our implementation is heavily built upon their codes.
