import pickle
import numpy as np
import pandas as pd
import torch

# 从 scipy.stats 中导入用于计算 推土机距离、KL散度 和 峰度 的函数
from scipy.stats import wasserstein_distance, entropy as kl_divergence, kurtosis

# 从 scipy.spatial.distance 中导入用于计算 余弦距离 的函数
from scipy.spatial.distance import cosine as cosine_distance

print("所有必要的库都已成功导入！")
print("指纹提取器已准备就绪。")
# =============================================================================
# 步骤二：加载成对的热力图数据
# =============================================================================
try:
    with open('paired_heatmaps.pkl', 'rb') as f:
        paired_heatmaps = pickle.load(f)
    print(f"\n成功加载 {len(paired_heatmaps)} 组配对的热力图数据。")

except FileNotFoundError:
    print("\n错误：找不到 paired_heatmaps.pkl 文件。")
    print("请确保您已经成功运行了 'generate_heatmaps.py' 脚本，并且生成的文件与当前脚本在同一个文件夹下。")
    # 如果找不到文件，则退出程序，防止后续代码报错
    exit()