import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# 导入 Captum 库中的核心模块：LRP 和可视化工具
from captum.attr import LRP
from captum.attr import visualization as viz

print("所有库导入成功！")

# =============================================================================
# 第1部分：基础设置 (修正部分)
# --- 我们把 device 的定义提前到这里 ---
# =============================================================================
# 设置设备（如果电脑有NVIDIA显卡，则使用GPU，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")


# =============================================================================
# 第2部分：加载模型和数据
# =============================================================================
# --- 加载模型 ---
model = models.resnet18(weights=None)
model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
print(f"正在从本地路径加载预训练模型: {model_path}")

try:
    # --- 现在使用 device 变量时，它已经被定义了 ---
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
except FileNotFoundError:
    print(f"错误: 在路径 {model_path} 未找到权重文件。")
    exit()

model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)
model.eval()
print("\n模型已通过您的原始方法加载成功！")

# --- 加载漏洞数据 ---
try:
    with open('all_vulnerabilities.pkl', 'rb') as f:
        all_vulnerabilities = pickle.load(f)
    print(f"成功加载 {len(all_vulnerabilities)} 个漏洞样本！")

    if all_vulnerabilities:
        vulnerability_sample = all_vulnerabilities[0]
        print("\n已选中第一个漏洞样本进行处理。")
    else:
        print("错误：漏洞列表中没有样本。请先运行 generate_vulnerabilities.py")
        exit()
except FileNotFoundError:
    print("错误：找不到 all_vulnerabilities.pkl 文件。")
    exit()


# =============================================================================
# 第3部分：初始化LRP并计算归因
# =============================================================================
# 1. 准备LRP需要的输入数据
adversarial_image = vulnerability_sample['adversarial_image'].to(device)
target_class = int(vulnerability_sample['adversarial_pred'])
original_image = vulnerability_sample['original_image'].to(device)
true_label = vulnerability_sample['label']

print("\n--- 开始计算LRP归因 ---")
print(f"分析目标: 对抗样本 (真实类别: {true_label}, 模型错误预测为: {target_class})")

# 2. 初始化LRP分析器
lrp = LRP(model)

# 3. 计算归因（即生成热力图数据）
attribution = lrp.attribute(adversarial_image.unsqueeze(0), target=target_class)
print(f"归因计算完成！热力图数据的形状: {attribution.shape}")


# =============================================================================
# 第4部分：可视化归因热力图
# =============================================================================
print("\n--- 开始生成可视化图像 ---")

original_img_np = original_image.cpu().detach().permute(1, 2, 0).numpy()
adversarial_img_np = adversarial_image.cpu().detach().permute(1, 2, 0).numpy()

plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 显示原始图片
axes[0].imshow(original_img_np)
axes[0].set_title(f'Original Image\nTrue Label: {true_label}')
axes[0].axis('off')

# 2. 显示对抗样本图片
axes[1].imshow(adversarial_img_np)
axes[1].set_title(f'Adversarial Image\nPredicted: {target_class}')
axes[1].axis('off')

# 3. 显示LRP热力图
attribution_np = attribution.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
viz.visualize_image_attr(
    attr=attribution_np,
    original_image=adversarial_img_np,
    method='blended_heat_map',
    sign='all',
    show_colorbar=True,
    title='LRP Heatmap',
    fig=fig,
    axes=axes[2],
    use_pyplot=False
)
axes[2].axis('off')

plt.tight_layout()
print("图像已生成，正在显示...")
plt.show()

print("\n脚本执行完毕！")