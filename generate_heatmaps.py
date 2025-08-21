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

# --- 加载模型 ---
# (此部分已恢复到与您的 generate_vulnerabilities.py 完全相同的版本)

# 1. 实例化一个空的ResNet-18模型结构
model = resnet18(weights=None) 

# 2. 定义指向您本地缓存的权重文件的路径
#    这里我们直接使用您验证过的、能工作的代码
model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
print(f"正在从本地路径加载预训练模型: {model_path}")

# 3. 加载权重。strict=False 允许我们稍后替换fc层
try:
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
except FileNotFoundError:
    print(f"错误: 在路径 {model_path} 未找到权重文件。")
    print("请确认该文件存在，或者检查您的原始脚本以确认路径。")
    exit()

# 4. 替换最后一层以匹配SVHN的10个类别
model.fc = nn.Linear(model.fc.in_features, 10)

# 将模型移动到指定设备并设置为评估模式
model.to(device)
model.eval()

print("\n模型已通过您的原始方法加载成功！")


# --- 加载漏洞数据 ---
# (这一部分保持不变)
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
        exit()

except FileNotFoundError:
    print("错误：找不到 all_vulnerabilities.pkl 文件。")
    print("请确保您已经成功运行了 generate_vulnerabilities.py 脚本。")
    exit()