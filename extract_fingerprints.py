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
# =============================================================================
# 步骤三：实现五个核心的特征计算函数
# =============================================================================

def calculate_wasserstein(h1_tensor, h2_tensor):
    """
    计算两张热力图之间的推土机距离 (Wasserstein Distance)。

    Args:
        h1_tensor (torch.Tensor): 第一张热力图。
        h2_tensor (torch.Tensor): 第二张热力图。

    Returns:
        float: 两张热力图之间的推土机距离。
    """
    # 步骤 1: 将输入的PyTorch Tensor转换为NumPy数组
    h1_np = h1_tensor.detach().numpy()
    h2_np = h2_tensor.detach().numpy()

    # 步骤 2: 将二维的热力图矩阵展平（flatten）为一维向量
    h1_flat = h1_np.flatten()
    h2_flat = h2_np.flatten()

    # 步骤 3: 调用scipy函数计算并返回推土机距离
    distance = wasserstein_distance(h1_flat, h2_flat)
    
    return distance    

def calculate_cosine_similarity(h1_tensor, h2_tensor):
    """
    计算两张热力图之间的余弦相似度 (Cosine Similarity)。

    Args:
        h1_tensor (torch.Tensor): 第一张热力图。
        h2_tensor (torch.Tensor): 第二张热力图。

    Returns:
        float: 两张热力图之间的余弦相似度。
    """
    # 步骤 1: 使用 .detach() 将Tensor从计算图中分离，然后转换为NumPy数组
    h1_np = h1_tensor.detach().numpy()
    h2_np = h2_tensor.detach().numpy()

    # 步骤 2: 将二维的热力图矩阵展平为一维向量
    h1_flat = h1_np.flatten()
    h2_flat = h2_np.flatten()

    # 步骤 3: 计算余弦距离。注意：scipy计算的是“距离”，而不是“相似度”
    # 余弦距离 = 1 - 余弦相似度
    distance = cosine_distance(h1_flat, h2_flat)
    
    # 步骤 4: 将距离转换回我们需要的相似度
    similarity = 1 - distance
    
    return similarity
def _prepare_distributions(h_tensor):
    """
    辅助函数：将单个热力图处理成正、负两个子概率分布。
    """
    # 展平热力图
    h_flat = h_tensor.detach().flatten()
    
    # 步骤 1: 计算总相关性强度 (Total Relevance)
    # 这是我们统一的“货币单位”
    total_relevance = torch.sum(torch.abs(h_flat))
    epsilon = 1e-10 # 防止除以零

    # 步骤 2: 分离正、负贡献
    # torch.clamp(min=0) 会将所有负值变为0，保留所有正值
    h_pos = torch.clamp(h_flat, min=0)
    # torch.clamp(max=0) 会将所有正值变为0，保留所有负值
    # 我们再取绝对值，得到正数的负贡献强度
    h_neg = torch.abs(torch.clamp(h_flat, max=0))

    # 步骤 3: 按总强度归一化，得到两个子概率分布
    p_pos = h_pos / (total_relevance + epsilon)
    p_neg = h_neg / (total_relevance + epsilon)
    
    # 为了数值稳定性，给每个元素都加上极小值
    p_pos += epsilon
    p_neg += epsilon
    
    return p_pos.numpy(), p_neg.numpy()


def calculate_kl_divergences(h_clean_tensor, h_vuln_tensor):
    """
    计算干净热力图与失效热力图之间，正、负贡献分布的KL散度。
    
    Args:
        h_clean_tensor (torch.Tensor): 干净样本的热力图 (基准分布 P)。
        h_vuln_tensor (torch.Tensor): 失效样本的热力图 (近似分布 Q)。

    Returns:
        tuple[float, float]: 返回一个元组，包含 (正贡献KL散度, 负贡献KL散度)。
    """
    # 步骤 1: 为两张热力图分别准备正、负子概率分布
    p_clean_pos, p_clean_neg = _prepare_distributions(h_clean_tensor)
    p_vuln_pos, p_vuln_neg = _prepare_distributions(h_vuln_tensor)

    # 步骤 2: 分别计算正、负贡献的KL散度
    kl_pos = kl_divergence(p_clean_pos, p_vuln_pos)
    kl_neg = kl_divergence(p_clean_neg, p_vuln_neg)
    
    return kl_pos, kl_neg
def calculate_std_dev(h_vuln_tensor):
    """
    计算单张热力图（特指失效样本的）像素值的标准差。

    Args:
        h_vuln_tensor (torch.Tensor): 失效样本的热力图。

    Returns:
        float: 热力图像素值的标准差。
    """
    # 步骤 1: 将Tensor安全地转换为NumPy数组
    h_vuln_np = h_vuln_tensor.detach().numpy()
    
    # 步骤 2: 直接调用NumPy的std函数计算并返回标准差
    std_deviation = np.std(h_vuln_np)
    
    return std_deviation

# =============================================================================
# 步骤四：批量处理所有样本，提取指纹
# =============================================================================

# 创建一个列表，用于存储每个漏洞样本的最终指纹数据
fingerprints_list = []

print(f"\n--- 开始为 {len(paired_heatmaps)} 组热力图提取指纹 ---")

# 遍历我们从 .pkl 文件中加载的每一组成对的热力图数据
for i, data_pair in enumerate(paired_heatmaps):
    
    # 从数据对中取出干净热力图、失效热力图和漏洞类型标签
    h_clean = data_pair['h_clean']
    h_vuln = data_pair['h_vuln']
    vuln_type = data_pair['vulnerability_type']
    
    print(f"\r  正在处理样本 {i+1}/{len(paired_heatmaps)}", end="")
    
    # --- 调用我们之前定义的所有函数，计算6个特征值 ---
    
    # 1. 对比性特征
    wasserstein = calculate_wasserstein(h_clean, h_vuln)
    cosine_sim = calculate_cosine_similarity(h_clean, h_vuln)
    kl_pos, kl_neg = calculate_kl_divergences(h_clean, h_vuln)
    
    # 2. 内在性特征
    std = calculate_std_dev(h_vuln)
    kurt = calculate_kurtosis(h_vuln)
    
    # --- 将所有结果存入一个字典 ---
    fingerprint_data = {
        'wasserstein_dist': wasserstein,
        'cosine_similarity': cosine_sim,
        'kl_divergence_pos': kl_pos,
        'kl_divergence_neg': kl_neg,
        'std_dev': std,
        'kurtosis': kurt,
        'vulnerability_type': vuln_type
    }
    
    # 将这个样本的指纹字典添加到总列表中
    fingerprints_list.append(fingerprint_data)

print("\n--- 所有指纹已成功提取 ---")

# =============================================================================
# 步骤五：使用Pandas将结果保存为CSV文件
# =============================================================================

# 将包含所有字典的列表，转换为一个Pandas DataFrame
fingerprints_df = pd.DataFrame(fingerprints_list)

# 定义输出文件名
output_filename = 'vulnerability_fingerprints.csv'

# 将DataFrame保存为CSV文件，不包含行索引
fingerprints_df.to_csv(output_filename, index=False)

print(f"\n指纹数据已成功保存到: {output_filename}")
print("项目核心阶段已完成！您现在拥有了可用于训练机器学习模型的数据集。")