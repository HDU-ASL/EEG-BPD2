import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

save_name = "night2" 
pre = f"/home/hyx/test/BCML/memory/{save_name}_predicted_poses.txt"
tar = f"/home/hyx/test/BCML/memory/{save_name}_target_poses.txt"


save_dir = "/home/hyx/test/BCML/memory"

# pre = f"/home/hyx/test/BCML/memory_EEG/{save_name}_predicted_poses.txt"
# tar = f"/home/hyx/test/BCML/memory_EEG/{save_name}_target_poses.txt"



image_filename = osp.join(save_dir, f"{save_name}--2d.png")

gt_pose = np.loadtxt(tar)         # shape: [N, 7] - GT
pred_pose = np.loadtxt(pre)       # shape: [N, 7] - Predicted
assert gt_pose.shape[0] == pred_pose.shape[0], "预测和GT行数不一致"
assert gt_pose.shape[1] >= 3, "位置列数不足"

gt_xyz = gt_pose[:, :3]
pred_xyz = pred_pose[:, :3]

fig1, ax1 = plt.subplots(figsize=(12, 9.6))
# ax1.scatter(gt_pose[:, 0], gt_pose[:, 2],color='black', s=25)
# ax1.scatter(pred_pose[:, 0], pred_pose[:, 2], color='blue', s=50, alpha=0.5)

# for i in range(len(gt_pose)):
#     ax1.plot([gt_pose[i, 0], pred_pose[i, 0]], [gt_pose[i, 2], pred_pose[i, 2]],color='red', linestyle='-', linewidth=2, alpha=0.5)

# ax1.set_xlabel('X [m]', fontsize=30, labelpad=15)
# ax1.set_ylabel('Z [m]', fontsize=30, labelpad=15)
# ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
# ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax1.tick_params(axis='both', labelsize=25, width=1)
# ax1.grid()

ax1.scatter(gt_pose[:, 1], gt_pose[:, 0],color='black', s=25)
ax1.scatter(pred_pose[:, 1], pred_pose[:, 0], color='blue', s=50, alpha=0.5)

for i in range(len(gt_pose)):
    ax1.plot([gt_pose[i, 1], pred_pose[i, 1]], [gt_pose[i, 0], pred_pose[i, 0]],color='red', linestyle='-', linewidth=2, alpha=0.5)
# 标签字体
ax1.set_xlabel('X [m]', fontsize=30, labelpad=20)
ax1.set_ylabel('Y [m]', fontsize=30, labelpad=20)
# 减少刻度数量
ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax1.tick_params(axis='both', labelsize=30, width=1)
# 去除网格
ax1.grid()
# ax1.plot(gt_pose[:, 0], gt_pose[:, 2], color='black', marker='o', linestyle='', label='Ground Truth Pose')
# ax1.plot(pred_pose[:, 0], pred_pose[:, 2], color='red', marker='s', linestyle='', label='Predicted Pose')
# for i in range(len(gt_pose)):
#     ax1.plot([gt_pose[i, 0], pred_pose[i, 0]], [gt_pose[i, 2], pred_pose[i, 2]], color='gray', linestyle='--', alpha=0.5)
# ax1.set_xlabel('x [m]')
# ax1.set_ylabel('y [m]')
# ax1.legend()
# ax1.grid(True)
# ax1.plot(gt_pose[:, 1], gt_pose[:, 0], color='black', marker='o', linestyle='', label='Ground Truth Pose')  
# ax1.plot(pred_pose[:, 1], pred_pose[:, 0], color='red', marker='s', linestyle='', label='Predicted Pose (predeg_poses)')  
# for i in range(len(gt_pose)): 
#     ax1.plot([gt_pose[i, 1], pred_pose[i, 1]], [gt_pose[i, 0], pred_pose[i, 0]], color='gray', linestyle='--', alpha=0.5)  
# ax1.set_xlabel('x [m]')  
# ax1.set_ylabel('y [m]')  
# ax1.legend()  
# ax1.grid(True)  


fig1.savefig(image_filename)
plt.close()

# Print output
print(f"3D plot saved to: {image_filename}")