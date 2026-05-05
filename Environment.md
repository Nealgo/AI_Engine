# 实验环境 (Experimental Environment)

## 1. 硬件环境 (Hardware Environment)
- **GPU**: NVIDIA GeForce RTX 3090
- **显存 (VRAM)**: 24 GB

## 2. 软件环境 (Software Environment)
- **操作系统 (OS)**: Linux (Ubuntu)
- **CUDA 版本**: 12.8
- **NVIDIA 驱动版本**: 570.124.04
- **Python 环境**: 建议使用 Python 3.10+ (根据编译缓存 `__pycache__/*cpython-310*` 推测)

## 3. 主要依赖库 (Python Dependencies)
本项目使用的核心 Python 库及其版本如下 (详见 `requirements.txt`)：
- `torch` == 2.2.1
- `torchvision` == 0.17.1
- `pytorch-wavelets` == 1.3.0
- `PyWavelets` == 1.5.0
- `pillow` == 10.2.0
- `numpy` == 1.26.4
- `scikit-image` == 0.22.0
- `tqdm` == 4.66.2
