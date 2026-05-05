# AI_Engine / WDNet

本项目包含一个基于深度学习的图像去雨（Image Deraining）算法模型。项目中实现了相关网络结构（WDNet）、数据加载机制以及训练和测试流水线。

## 项目结构 (Project Structure)

```text
.
├── datas/                  # 存放训练集和测试集数据 (由 organize.sh 整理)
├── imgs/                   # 存放一些示例图像或测试输出图像
├── model/                  # 核心模型定义目录
│   └── wdnet.py            # WDNet 模型网络结构
├── scripts/                # 脚本目录
│   └── organize.sh         # 数据集整理脚本 (将下载的 derain_drop_dataset 整理为标准格式)
├── dataLoader.py           # 数据加载与预处理逻辑
├── main.py                 # 主执行程序
├── train.py                # 模型训练脚本
├── test.py                 # 模型测试/推理脚本
├── requirements.txt        # Python 依赖清单
├── Environment.md          # 详细的实验环境配置说明
└── README.md               # 本项目说明文档
```

## 环境配置 (Environment Setup)

请参考详细实验环境文档：[Environment.md](./Environment.md)。

使用以下命令安装 Python 依赖：
```bash
pip install -r requirements.txt
```

## 数据准备 (Data Preparation)

本项目使用了 `derain_drop_dataset`。请运行 `scripts/organize.sh` 脚本来自动整理数据集的目录结构：
```bash
bash scripts/organize.sh
```
执行完毕后，数据将按要求存放在 `datas/archive/` 中的 `data` 和 `gt` 子目录下。

## 使用说明 (Usage)

### 训练模型 (Training)
可以运行以下命令启动训练过程（训练日志将自动保存到 csv 文件中）：
```bash
python train.py
# 或使用主程序启动
# python main.py
```

### 测试模型 (Testing)
使用已经训练好的模型权重对测试集或单张图像进行去雨测试：
```bash
python test.py
```
