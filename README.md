# DDPM 光学-SAR图像生成项目

## 项目概述

本项目采用扩散概率模型 (DDPM) 实现光学图像与合成孔径雷达 (SAR) 图像的配对生成。通过多个版本的迭代优化，实现了一种融合光学和SAR特征的深层架构，用于高质量图像生成。

## 项目特点

- **多分支架构**：独立提取光学和SAR特征，实现深度融合
- **选择性核融合** (SKFusion)：增强SAR特征与光学特征的交互
- **梯度累积**：支持大批量训练的梯度积累机制
- **学习率策略**：采用预热 + 余弦衰减的学习率调度
- **辅助损失**：融合L1损失提升图像质量

## 文件描述

### 核心训练脚本

| 文件 | 版本 | 主要特性 |
|------|------|---------|
| `sen_new0.py` | v0 | 基础DDPM实现，双分支PairDataset |
| `sen_new1.py` | v1 | 双分支架构：光学主干 + SAR辅助分支，多层特征融合 |
| `sen_new2.py` | v2 | 添加SKFusion选择性核融合模块，增强特征融合能力 |
| `sen_new3.py` | v3 | 优化学习率策略：预热 + 余弦衰减调度 |
| `sen_new4.py` | v4 | **最新版本**：改进梯度累积、优化日志、添加L1辅助损失 |

### 数据处理

| 文件 | 功能 |
|------|------|
| `data_revise.py` | 数据预处理和分割工具，支持配对图像的匹配、验证和train/val划分 |

## 安装依赖

```bash
# 创建并激活conda环境
conda create -n torch python=3.9
conda activate torch

# 安装必要的包
pip install torch torchvision
pip install einops tqdm pillow tensorboard
```

## 使用方法

### 1. 数据准备

使用 `data_revise.py` 处理和分割数据集：

```bash
python data_revise.py --src <原始数据路径> --dst <输出路径> --val-ratio 0.05
```

**数据结构要求**：
```
data/
├── category1/
│   ├── s1/           # SAR图像
│   │   ├── img_s1_.png
│   │   └── ...
│   └── s2/           # 光学图像
│       ├── img_s2_.png
│       └── ...
├── category2/
│   ├── s1/
│   └── s2/
└── ...
```

### 2. 模型训练

选择相应版本进行训练。推荐使用最新的 `sen_new4.py`：

```bash
python sen_new4.py \
    --data-path <数据集路径> \
    --batch-size 8 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --accumulation-steps 4
```

**主要参数说明**：
- `--data-path`: 训练数据集路径
- `--batch-size`: 批次大小
- `--epochs`: 训练轮数
- `--learning-rate`: 初始学习率
- `--accumulation-steps`: 梯度累积步数
- `--device`: 训练设备 (cuda/cpu)

### 3. 训练监控

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir=./logs
```

## 核心模块说明

### PairDataset

加载配对的光学-SAR图像，支持数据增强：

```python
dataset = PairDataset(
    root_dir='path/to/data',
    transform=transforms.Compose([...]),  # 光学图像变换
    L_transform=transforms.Compose([...])  # SAR图像变换
)
```

### 双分支UNet架构

- **光学主干网络**：提取光学图像的精细特征
- **SAR分支**：
  - 双路径设计（大核+小核卷积）
  - 可选的SKFusion融合模块
  - 多层注入到光学主干

### 学习率调度 (sen_new3+)

采用带预热的余弦衰减策略：
- 预热步数：总步数的5%
- 余弦衰减：从预热完成后开始
- 加速模型收敛，提高训练稳定性

### 损失函数优化 (sen_new4)

- **主损失**：MSE损失 (扩散模型标准)
- **辅助损失**：L1损失，帮助捕捉SAR图像结构
- **梯度累积**：先除累积步数再反向传播

## 版本演变总结

```
sen_new0: 基础框架
    ↓
sen_new1: 双分支融合架构
    ↓
sen_new2: SKFusion选择性核融合
    ↓
sen_new3: 学习率预热+余弦衰减
    ↓
sen_new4: 梯度累积+L1辅助损失 ⭐ 推荐使用
```

## 硬件要求

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060/4060 或更高)
- **CPU**: 4核+
- **内存**: 16GB+
- **存储**: 根据数据集大小调整 (建议50GB+)

## 输出文件

训练过程中生成：
- `checkpoints/`: 模型检查点
- `logs/`: TensorBoard日志
- `outputs/`: 生成的样本图像

## 常见问题

### Q: 如何调整模型复杂度？
A: 修改UNet的channel倍增系数和depth参数

### Q: 训练过程中显存不足？
A: 降低batch_size或增加accumulation_steps

### Q: 如何使用自己的数据？
A: 按照数据结构要求组织数据，然后使用data_revise.py处理

## 引用

如果本项目对你有帮助，请引用：

```
@article{ddpm2020,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2006.11239},
  year={2020}
}
```

## 许可证

MIT License

## 联系方式

有任何问题或建议，欢迎提出Issue和Pull Request。

---

**最后更新**: 2026年3月28日
