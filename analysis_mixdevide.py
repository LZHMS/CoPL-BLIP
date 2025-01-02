import re
import matplotlib.pyplot as plt
import os
import numpy as np

log_file_path = '/data1/zhli/dpl/output/analysis/dtd/rn200_16shots_12FP_pairflip/nctx16_cscFalse_ctpend_gceFalse/seed1/log.txt'

path_parts = log_file_path.split(os.sep)
dataset_name = path_parts[-5]
path_figure = path_parts[-4].split("_")[-2] + "_" + path_parts[-4].split("_")[-1]

figure_dir = os.path.join('/data1/zhli/dpl/analysis/mixdevide', dataset_name)
os.makedirs(figure_dir, exist_ok=True)

def analysis_midevide(path):
    patterns = {
        "Clean": r"Clean ratio in Clean set: ([0-9.]+)",
        "Noisy": r"Clean ratio in Noisy set: ([0-9.]+)",
        "Obscure": r"Clean ratio in Confused set: ([0-9.]+)"
    }

    ratios_dict = {key: [] for key in patterns.keys()}

    with open(path, 'r') as file:
        for line in file:
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    ratio_value = float(match.group(1))
                    ratios_dict[key].append(ratio_value)

    # 生成奇数序列，从1到199（因为您有100个数据点）
    x_values = np.arange(1, 201, 2)

    plt.figure(figsize=(12, 6))
    for key, ratios in ratios_dict.items():
        # 过滤掉值为0的数据点
        filtered_ratios = [ratio for ratio in ratios if ratio != 0]
        filtered_x_values = x_values[:len(filtered_ratios)]

        plt.plot(filtered_x_values, filtered_ratios, marker='.', label=f"{key} set")

    plt.xlim(0, 200)  
    plt.xticks(np.arange(0, 200, 20))  

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Clean Sample Ratio", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    figure_path = os.path.join(figure_dir, f"{path_figure}.png")
    plt.savefig(figure_path, dpi=300)
    plt.show()

analysis_midevide(log_file_path)