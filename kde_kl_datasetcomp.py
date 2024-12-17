import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import warnings

# 忽略可能的运行时警告
warnings.filterwarnings("ignore")

# 计算带宽 - Silverman's 规则
def silverman_bandwidth(data):
    """
    根据 Silverman's 规则计算核密度估计的带宽。
    :param data: 输入数据 (numpy array)
    :return: 计算的带宽值
    """
    std_dev = np.std(data, ddof=1)  # 数据的标准差
    n = len(data)                   # 数据样本数量
    bandwidth = 1.06 * std_dev * n**(-1/5)
    return bandwidth

# 核密度估计函数
def compute_kde(data, bandwidth='silverman'):
    """
    计算数据的核密度估计。
    :param data: 输入数据 (numpy array)
    :param bandwidth: 带宽 ('silverman', 'scott' 或具体的带宽因子)
    :return: 核密度估计对象
    """
    if bandwidth == 'silverman':
        bandwidth = silverman_bandwidth(data)

    kde = gaussian_kde(data, bw_method=bandwidth)
    return kde

# KL散度计算函数
def compute_kl_divergence(kde1, kde2, xmin, xmax, num_points=None):
    """
    计算两个核密度估计之间的KL散度。
    :param kde1: 第一个核密度估计对象
    :param kde2: 第二个核密度估计对象
    :param xmin: 估计范围的最小值
    :param xmax: 估计范围的最大值
    :param num_points: 用于积分的点数，如果为None，则使用KDE1的数据点数
    :return: KL散度值
    """
    if num_points is None:
        num_points = len(kde1.dataset[0])  # 使用KDE对象中的数据量作为num_points

    x = np.linspace(xmin, xmax, num_points)
    p = kde1(x)
    q = kde2(x)

    # 添加一个小的常数以避免零值
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    # 计算 p/q 并限制其最大值以防止溢出
    ratio = p / q
    max_ratio = 1e10  # 根据需要调整这个值
    ratio = np.clip(ratio, None, max_ratio)

    # 计算 KL 散度
    kl_div = np.sum(p * np.log(ratio)) * (xmax - xmin) / num_points

    # 检查是否有无穷大或NaN值
    if not np.isfinite(kl_div):
        print("警告：计算出的 KL 散度包含非有限值。")
        kl_div = np.inf  # 或者根据需求处理

    return kl_div

# 判断数据收敛的关键函数
def determine_data_sufficiency(data, epsilon=1e-4, increment=100, bandwidth='silverman'):
    """
    判断数据是否足够通过KL散度收敛。
    :param data: 输入数据 (numpy array)
    :param epsilon: KL散度收敛阈值
    :param increment: 每次增加的数据量
    :param bandwidth: KDE的带宽设置 ('silverman', 'scott' 或具体值)
    :return: 收敛时的数据量
    """
    n = len(data)
    for i in range(increment, n, increment):
        current_end = i + increment
        if current_end > n:
            current_end = n
        kde1 = compute_kde(data[:i], bandwidth)
        kde2 = compute_kde(data[:current_end], bandwidth)
        xmin, xmax = min(data[:i]), max(data[:i])

        # 生成x范围上的KDE结果，num_points自动设置为当前数据量
        # num_points = i  # 使用当前数据量作为num_points
        x = np.linspace(xmin, xmax, n)
        p = kde1(x)
        q = kde2(x)

        # 归一化KDE结果
        dx = (xmax - xmin) / len(x)
        p_normalized = p / (np.sum(p) * dx)
        q_normalized = q / (np.sum(q) * dx)

        # 计算KL散度
        p_normalized = np.where(p_normalized > 0, p_normalized, 1e-10)
        q_normalized = np.where(q_normalized > 0, q_normalized, 1e-10)
        kl_div = np.sum(p_normalized * np.log(p_normalized / q_normalized)) * dx

        # 判断是否收敛
        if kl_div < epsilon:
            return i

    return n

def process_feature(feature_name, feature_data, epsilon, increment, bandwidth):
    """
    处理单个特征，计算其足够数据量。
    :param feature_name: 特征名称 (str)
    :param feature_data: 特征数据 (numpy array)
    :param epsilon: KL散度收敛阈值
    :param increment: 每次增加的数据量
    :param bandwidth: KDE的带宽设置
    :return: (feature_name, sufficient_data_amount)
    """
    sufficient = determine_data_sufficiency(feature_data, epsilon=epsilon, increment=increment, bandwidth=bandwidth)
    return (feature_name, sufficient)

# 示例：鼠标序列数据的分析

def main():
    # 在这里读数据===========
    # 读取CSV文件
    data = pd.read_csv('D:/datauser/Data_files/user23/session_0405064924.csv')
    print(f"数据总行数: {len(data)}")

    # 指定要读取的列，请根据实际的列名替换下面的列名
    features = {
        'button': data['button'].values,
        'state': data['state'].values,
        'x': data['x'].values,
        'y': data['y'].values,
        'distance': data['distance'].values,
        'velocity': data['velocity'].values,
        'acceleration': data['acceleration'].values,
        'curvature': data['curvature'].values,
        'angle_change': data['angle_change'].values,
        'x_velocity': data['x_velocity'].values,
        'y_velocity': data['y_velocity'].values,
        'x_acceleration': data['x_acceleration'].values,
        'y_acceleration': data['y_acceleration'].values,
        'press_duration': data['press_duration'].values
    }
    #===========

    # 分别对每个特征判断数据收敛量
    # 设置参数
    increment = 200
    epsilon = 1e-4
    bandwidth = 'silverman'

    # 获取CPU核心数量
    num_cores = multiprocessing.cpu_count()

    print(f"使用 {num_cores} 核心进行并行处理...")

    # 使用 joblib 进行并行处理
    results = Parallel(n_jobs=num_cores, prefer="threads")(
        delayed(process_feature)(name, features[name], epsilon, increment, bandwidth) for name in features
    )

    # 打印结果
    for feature_name, sufficient_amount in results:
        print(f"{feature_name} 的足够数据量: {sufficient_amount}")

if __name__ == "__main__":
    main()
