##使用 PyTorch 进行 CIFAR-10 图像分类
##本项目演示了如何使用 PyTorch 训练一个神经网络来对 CIFAR-10 数据集中的图像进行分类。该模型能够将图像分类为 10 个不同的类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。代码支持多种网络架构，如 ResNet、VGG、DenseNet 等。
##目录 
	•   要求
	•   安装

	•   使用方法

	•   模型架构

	•   训练
	
	•   测试

	•   保存检查点

	•   许可协议
##要求
•   Python 3.6+
•   PyTorch
•   torchvision
•   CUDA（可选，用于 GPU 加速）

##安装
1.克隆代码仓库：
git clone https://github.com/your-repo/cifar10-pytorch.git
cd cifar10-pytorch
2.安装所需的依赖：
pip install -r requirements.txt
3.（可选）如果计划使用 GPU，请确保已安装 CUDA 和 cuDNN。

##使用方法
1. 训练模型
要使用默认设置训练模型，请运行：
    python main.py --lr 0.1
您可以指定不同的学习率或从检查点恢复训练，方法是使用 --resume 参数：
    python main.py --resume
	
2. 模型架构
脚本支持多种模型架构。您可以通过取消注释 main.py 脚本中的相应行来选择其中的任何一种：
•   VGG19
•   ResNet18
•   PreActResNet18
•   GoogLeNet
•   DenseNet121
•   ResNeXt29_2x64d
•   MobileNet / MobileNetV2
•   DPN92
•   ShuffleNetG2 / ShuffleNetV2
•   SENet18
•   EfficientNetB0
•   RegNetX_200MF
•   SimpleDLA（默认）


##您可以在以下行中替换模型：
net = SimpleDLA()  # 替换为您选择的模型

3. 训练
train 函数负责一个训练周期。它：
•   从 CIFAR-10 数据集中加载图像批次。
•   将它们传递通过选择的神经网络。
•   使用交叉熵计算损失。
•   通过反向传播损失并更新网络参数。
训练细节（损失、准确率）通过进度条在训练期间显示。

4. 测试
test 函数评估模型在测试集上的表现。它：
•   将测试图像通过网络。
•   计算损失和准确率。
•   如果准确率提高，则保存模型检查点。

5. 保存检查点
在测试期间，如果准确率提高，则模型的状态会保存为检查点，存储在 ./checkpoint/ 目录中。您可以通过运行带有 --resume 标志的脚本从最后一个检查点恢复训练。
python main.py --resume

6. 学习率调度器
使用余弦退火学习率调度器来调整训练期间的学习率。它会在训练过程中将学习率从初始值逐渐减少到 0。

##CIFAR-10 数据集

CIFAR-10 数据集包含 60,000 张 32x32 彩色图像，分为 10 类，每类 6,000 张图像。数据集分为 50,000 张训练图像和 10,000 张测试图像。如果指定的目录中不存在数据集，它将自动下载。
 • 训练时的转换：应用随机裁剪和水平翻转来增强数据。
 • 测试时的转换：将图像归一化为 CIFAR-10 数据集的均值和标准差。

许可协议 本项目根据 MIT 许可协议进行许可。


## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

