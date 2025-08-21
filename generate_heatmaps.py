import torch
import torch.nn as nn
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