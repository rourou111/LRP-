import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt
import numpy as np

# 导入 Captum 库中的核心模块：LRP 和可视化工具
from captum.attr import LRP
from captum.attr import visualization as viz

print("所有库导入成功！")

# 设置设备（如果电脑有NVIDIA显卡，则使用GPU，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")

# 1. 实例化一个在ImageNet上预训练的ResNet-18模型结构
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# 2. 将模型的最后一层（全连接层）替换为适用于SVHN的10分类层
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 将模型移动到指定的设备（CPU或GPU）
model.to(device)
# !!! 关键一步：将模型设置为评估模式 !!!
model.eval()

print("\n模型加载成功！（已适配SVHN数据集）")


# --- 加载漏洞数据 ---
# 这一部分无需修改
try:
    with open('all_vulnerabilities.pkl', 'rb') as f:
        all_vulnerabilities = pickle.load(f)
    print(f"成功加载 {len(all_vulnerabilities)} 个漏洞样本！")

    # 为了调试和验证，我们先只处理第一个漏洞样本
    if all_vulnerabilities:
        vulnerability_sample = all_vulnerabilities[0]
        print("\n已选中第一个漏洞样本进行处理。")
    else:
        print("错误：漏洞列表中没有样本。请先运行 generate_vulnerabilities.py")
        exit() # 如果没有样本，则退出程序

except FileNotFoundError:
    print("错误：找不到 all_vulnerabilities.pkl 文件。")
    print("请确保您已经成功运行了 generate_vulnerabilities.py 脚本，并且该文件与当前脚本在同一个文件夹下。")
    exit() # 如果找不到文件，则退出程序