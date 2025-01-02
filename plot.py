import re
import matplotlib.pyplot as plt
import os
import numpy as np


# log_file_path = '/data1/zhli/dpl/output/base_200ep_3seed_new/oxford_flowers/rn200_16shots_10FP_pairflip/nctx16_cscFalse_ctpend_gceFalse/seed1/log.txt'
log_file_path = '/data1/zhli/dpl/output/base_200ep_3seed_new/oxford_flowers/rn200_16shots_12FP_pairflip/nctx16_cscFalse_ctpend_gceFalse/seed1/log.txt'

log_dir_path = os.path.dirname(log_file_path)


path_parts = log_file_path.split(os.sep)
dataset_name = path_parts[-5]
path_figure = path_parts[-4].split("_")[-2] + "_" + path_parts[-4].split("_")[-1]

figure_dir = os.path.join('/data1/zhli/dpl/figure', dataset_name)
os.makedirs(figure_dir, exist_ok=True)


batches = []
accuracies = []
x_values = [] 


with open(log_file_path, 'r') as file:
    for line in file:
        
        epoch_match = re.search(r'epoch \[(\d+)/\d+\]', line)
        batch_match = re.search(r'batch \[(\d+)/\d+\]', line)
        # acc_match = re.search(r'acc (\d+\.\d+)', line)
        acc_match = re.search(r'accuracy: (\d+\.\d+)%', line)
        
     
        if epoch_match and batch_match and acc_match:
            epoch = int(epoch_match.group(1))
            batch = int(batch_match.group(1))
            if acc_match:
                acc = float(acc_match.group(1))
                print(acc)  # 输出: 62.3
            else:
                print("No match found")

            
            batches.append(batch)
            accuracies.append(acc)
            x_values.append((epoch - 1) * 50 + batch)  


plt.plot(x_values, accuracies, label='acc')

plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Batch vs Accuracy')
plt.legend()


figure_path = os.path.join(figure_dir, f"{path_figure}.png")


plt.savefig(figure_path)


plt.close()