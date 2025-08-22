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
# --- 初始化LRP并计算归因 ---

# 1. 准备LRP需要的输入数据
# 从我们加载的样本字典中，取出对抗样本图片
adversarial_image = vulnerability_sample['adversarial_image'].to(device)
# 取出这张图片被模型错误识别的类别ID
target_class = vulnerability_sample['adversarial_pred']
# 取出原始的、未被攻击的图片，用于后续对比
original_image = vulnerability_sample['original_image'].to(device)
# 取出图片本来的、正确的类别ID
true_label = vulnerability_sample['label']

print("\n--- 开始计算LRP归因 ---")
print(f"分析目标: 对抗样本 (真实类别: {true_label}, 模型错误预测为: {target_class})")


# 2. 初始化LRP分析器
# 我们直接用已经加载并设置为评估模式的模型来创建一个LRP对象
lrp = LRP(model)


# 3. 计算归因（即生成热力图数据）
# 调用 attribute 方法，这是Captum的核心功能
# 输入: 
#   - adversarial_image.unsqueeze(0): 我们的图片需要增加一个“批次”维度，这是PyTorch模型的标准输入格式 (C, H, W) -> (N, C, H, W)
#   - target=target_class: 我们告诉LRP，请分析模型为什么会预测出这个“错误的目标类别”
attribution = lrp.attribute(adversarial_image.unsqueeze(0), target=target_class)

# attribution 变量现在包含了每个输入像素对最终错误决策的“贡献分数”
# 它的形状和原始图片是一样的
print(f"归因计算完成！热力图数据的形状: {attribution.shape}")

# --- 可视化归因热力图 ---
print("\n--- 开始生成可视化图像 ---")

# 准备工作：PyTorch的图像张量是 (C, H, W)，即 (通道, 高, 宽)
# 而 Matplotlib 显示图像需要 (H, W, C)，我们需要转换一下维度
original_img_np = original_image.cpu().permute(1, 2, 0).numpy()
adversarial_img_np = adversarial_image.cpu().permute(1, 2, 0).numpy()

# 清理一下之前可能存在的图像
plt.close('all')

# 创建一个1行3列的画布
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
# 我们使用 captum 自带的可视化工具，它功能非常强大
# a. 先处理归因数据的维度，使其符合可视化工具的要求
attribution_np = attribution.squeeze(0).cpu().permute(1, 2, 0).numpy()
# b. 调用可视化函数
viz.visualize_image_attr(
    attr=attribution_np,
    original_image=adversarial_img_np,
    method='blended_heat_map',
    sign='all',  # 'all', 'positive', or 'negative'
    show_colorbar=True,
    title='LRP Heatmap',
    fig=fig,
    axes=axes[2],
    use_pyplot=False # 我们手动管理图像显示
)
axes[2].axis('off') # 关闭坐标轴

# 调整布局并显示图像
plt.tight_layout()
print("图像已生成，正在显示...")
plt.show()

print("\n脚本执行完毕！")