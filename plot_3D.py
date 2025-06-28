import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

# 文件路径
save_name = "sphere"  # 或使用 self.name
pre = f"/home/hyx/test/BCML/memory/{save_name}_predicted_poses.txt"
tar = f"/home/hyx/test/BCML/memory/{save_name}_target_poses.txt"


# 保存路径及图像命名（你可以替换 self.name 为具体名称）
save_dir = "/home/hyx/test/BCML/memory"

# pre = f"/home/hyx/test/BCML/memory_EEG/{save_name}_predicted_poses.txt"
# tar = f"/home/hyx/test/BCML/memory_EEG/{save_name}_target_poses.txt"


# # 保存路径及图像命名（你可以替换 self.name 为具体名称）
# save_dir = "/home/hyx/test/BCML/memory_EEG"
index_file = "/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv"
image_filename = osp.join(save_dir, f"{save_name}-3d.png")
gt_pose = np.loadtxt(tar)
pred_pose = np.loadtxt(pre)
sorted_indices = np.loadtxt(index_file, dtype=int)

# 排序

gt_pose = np.loadtxt(tar)         # shape: [N, 7] - GT
pred_pose = np.loadtxt(pre)       # shape: [N, 7] - Predicted
sort_order = np.argsort(sorted_indices)
gt_pose = gt_pose[sort_order]
pred_pose = pred_pose[sort_order]

# 加载位姿数据
# 检查数据一致性
assert gt_pose.shape[0] == pred_pose.shape[0], "预测和GT行数不一致"
assert gt_pose.shape[1] >= 3, "位置列数不足"

# 提取位置 (x, y)
gt_xyz = gt_pose[:, :3]
pred_xyz = pred_pose[:, :3]

# Create a 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color='gray', linewidth=3)
ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color='black', s=25)
ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color='blue', s=50, alpha=0.5)

for i in range(len(gt_xyz)):
    ax.plot([gt_xyz[i, 0], pred_xyz[i, 0]],
            [gt_xyz[i, 1], pred_xyz[i, 1]],
            [gt_xyz[i, 2], pred_xyz[i, 2]],
            color='red', linestyle='-', linewidth=3, alpha=0.85)


# ax.plot(gt_xyz[:, 1], gt_xyz[:, 0], gt_xyz[:, 2], color='gray', linewidth=3)
# ax.scatter(gt_xyz[:, 1], gt_xyz[:, 0], gt_xyz[:, 2], color='black', s=25)
# ax.scatter(pred_xyz[:, 1], pred_xyz[:, 0], pred_xyz[:, 2], color='blue', s=50, alpha=0.5)

# for i in range(len(gt_xyz)):
#     ax.plot([gt_xyz[i, 1], pred_xyz[i, 1]],
#             [gt_xyz[i, 0], pred_xyz[i, 0]],
#             [gt_xyz[i, 2], pred_xyz[i, 2]],
#             color='crimson', linestyle='--', linewidth=3, alpha=0.65)

ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.set_xlabel('X [m]', fontsize=30, labelpad=25)
ax.set_ylabel('Y [m]', fontsize=30, labelpad=25)
ax.set_zlabel('Z [m]', fontsize=30, labelpad=22)
from matplotlib.ticker import MaxNLocator

ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
ax.tick_params(axis='both', labelsize=26, width=1)
for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    label.set_fontweight('bold')
ax.grid(True, linestyle='-', linewidth=0.3)
ax.set_facecolor('whitesmoke')  # Light background color



fig.savefig(image_filename)
plt.close()

# Print output
print(f"3D plot saved to: {image_filename}")