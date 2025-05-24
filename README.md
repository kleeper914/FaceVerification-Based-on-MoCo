本项目要实现一个基于[MoCo](https://arxiv.org/abs/1911.05722)的自监督学习人脸预训练框架，并结合semi-hard triplet策略进行微调优化，最终在LFW和自爬数据上评估人脸验证性能

## 📁项目结构

~~~bash
FaceVerification Based On MoCo/
├── data/              # 数据文件夹
├── dataset.py        # 定义数据集类与 semi-hard triplet 构造
├── evaluate.py       # 模型评估逻辑脚本
├── losses.py         # 定义损失函数
├── manager.py        # 管理训练流程、日志
├── moco/              # MoCo 模块核心结构目录
│   ├── loader.py         # MoCo 数据加载器
│   └── model.py          # MoCo 模型定义
├── models.py         # 定义 backbone 网络，ResNet34/50、InceptionV1
├── train_moco.py     # 无监督对比训练主程序
└── train_triplet.py  # 使用 Triplet Loss 的监督训练主程序
~~~

## 🚀 功能概述

### 自监督表示学习：MoCo 训练

+ 使用MoCo v2框架对CASIA-Webface的15w张人脸图像进行表征学习
+ 预训练基于三种encoder：ResNet34/50、InceptionV1

### 验证阶段：LFW 上的相似度比对

+ 使用余弦相似度评估相似性
+ 根据AUC和Youden index自动搜索最优threshold

### 微调阶段：semi-hard triplets训练

+ 使用online search的策略，在训练阶段的每一个mini-batch 中构建(anchor, postive, semi-hard negative)样本对
+ 使用triplet loss微调MoCo预训练的模型，使得在人脸验证任务方面取得更优的效果

## 📦 安装依赖

建议python 版本 3.8+

~~~txt
torch>=1.12
torchvision>=0.13
numpy>=1.20
scikit-learn>=0.24
tqdm>=4.60
Pillow>=8.0
~~~

## 🧪数据准备

项目使用以下数据：

+ [CASIA-Webface](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface)
+ [LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
+ iMDb自爬人脸数据

## 🏃‍♂️ 训练与验证

您可以使用如下命令启动MoCo的训练

~~~bash
python train_moco.py /path/to/dataset \
 --model resnet50 \
 --seed 2025 \
 --epochs 200 \
 --batch_size 256 \
 --lr 0.03 \
 --momentum 0.9 \
 --weight-decay 1e-4 \
 --dim 128
~~~

您可以使用如下命令启动Triplet微调训练

~~~bash
python train_triplet.py /path/to/datset \
 --model resnet50 \
 --model-path /path/to/model_weights \
 --dim 128 \
 --epochs 100 \
 --batch_size 256 \
 --lr 0.01 \
 --weight-decay 1e-4 \
 --num-pairs 10000 \
 --margin 0.2
~~~

## 📊 训练结果

下述为MoCo预训练50epoch后的结果

| Encoder     | Accuracy | AUC   | F1-score |
| ----------- | -------- | ----- | -------- |
| ResNet-34   | 72.3%    | 0.776 | 0.701    |
| ResNet-50   | 75.1%    | 0.801 | 0.732    |
| InceptionV1 | 70.5%    | 0.762 | 0.684    |

下述为Triplet微调30epoch后的结果

| Encoder     | Accuracy | AUC   | F1-score |
| ----------- | -------- | ----- | -------- |
| ResNet-34   | 83.2%    | 0.867 | 0.802    |
| ResNet-50   | 85.1%    | 0.844 | 0.821    |
| InceptionV1 | 80.5%    | 0.832 | 0.792    |

