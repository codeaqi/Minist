# MNIST 手写数字识别

这是一个基于 PyTorch 的 MNIST 手写数字识别项目，保留了最原始的单文件脚本 [cnn_mnist_pytorch.py](./cnn_mnist_pytorch.py)，同时提供了一个按规范拆分后的新版本入口 [Minist_eval/train.py](./Minist_eval/train.py)。

## 项目结构

```text
.
├── cnn_mnist_pytorch.py        # 原始单文件版本
├── Minist_eval/                # 模块化版本
├── config/                     # YAML 配置
├── data/                       # MNIST 数据
├── modelpara.pth               # 预训练模型参数
└── README.md
```

## 运行方式

### 原始单文件版本

```powershell
python cnn_mnist_pytorch.py
```

### 模块化版本

```powershell
python -m Minist_eval.train --config config/v1.0_baseline_Scratch.yaml
```

## 配置说明

模块化版本的超参数集中在 [config/v1.0_baseline_Scratch.yaml](./config/v1.0_baseline_Scratch.yaml) 里，包括：

- 数据路径
- batch size
- 是否训练
- 是否使用 GPU
- 模型结构参数
- 优化器和学习率调度器

## 数据说明

项目默认使用 MNIST 数据集。当前仓库中已经包含了处理后的数据文件，模块化版本会优先读取本地数据。

## 模型说明

模型是一个简单的卷积神经网络，适合做 MNIST 入门示例：

- 两层卷积
- 两层池化
- 两层全连接
- 输出 10 类数字结果

## 当前效果

在当前配置下，测试集准确率约为 `99.22%`。

