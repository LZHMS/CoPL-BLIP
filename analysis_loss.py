import re
import matplotlib.pyplot as plt
import os
import numpy as np

def analysis_loss(log_file_paths, figure_dir, path_figure, dataset_name):
    """
    绘制两个log文件的曲线在同一张图上，并指定颜色和标签区分曲线

    Args:
        log_file_paths (list): 包含两个log文件路径的列表
        figure_dir (str): 保存图像的目录
        dataset_name (str): 数据集名称
    """
    # 检查输入文件数量是否为2
    if len(log_file_paths) != 2:
        raise ValueError("This function is designed to handle exactly two log files.")

    noisy_rates_list = []
    labels = ['Double Model', 'Simple Model']  
    colors = ['red', 'blue']  

    # 遍历所有log文件路径
    for log_file_path in log_file_paths:
        noisy_rates = []
        try:
            # 解析log文件
            with open(log_file_path, 'r') as file:
                for line in file:
                    match = re.search(r"refined noisy rate: ([0-9.]+)", line)
                    if match:
                        noisy_rate = float(match.group(1))
                        noisy_rates.append(noisy_rate)
            
            # 添加到结果列表
            noisy_rates_list.append(noisy_rates)
        
        except FileNotFoundError:
            print(f"The file {log_file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while processing {log_file_path}: {e}")

    # 开始绘图
    plt.figure(figsize=(12, 6))
    
    for i, noisy_rates in enumerate(noisy_rates_list):
        x_values = np.arange(len(noisy_rates))
        plt.plot(
            x_values, noisy_rates, 
            marker='o', linestyle='-', 
            color=colors[i], label=labels[i]
        )
    
    plt.xlim(0, max(len(rates) for rates in noisy_rates_list) - 1)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Noisy Rate", fontsize=12)
    plt.legend( fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    figure_path = os.path.join(figure_dir, f"{dataset_name}_{path_figure}_comparison.png")
    plt.savefig(figure_path, dpi=300)
    plt.show()

# Example usage:
log_file_paths = [
    '/data1/zhli/dpl/output/doublemodel/dtd/rn200_16shots_12FP_symflip/nctx16_cscFalse_ctpend_gceFalse/seed1/log.txt',
    '/data1/zhli/dpl/output/simplemodel/dtd/rn200_16shots_12FP_symflip/nctx16_cscFalse_ctpend_gceFalse/seed1/log.txt'
]
path_parts = log_file_paths[0].split(os.sep)
dataset_name = path_parts[-5]
figure_dir = os.path.join('/data1/zhli/dpl/analysis/model_loss', dataset_name)
path_figure = path_parts[-4].split("_")[-2] + "_" + path_parts[-4].split("_")[-1]
os.makedirs(figure_dir, exist_ok=True)

analysis_loss(log_file_paths, figure_dir, path_figure, dataset_name)
