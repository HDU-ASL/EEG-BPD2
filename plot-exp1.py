import numpy as np
import matplotlib.pyplot as plt
import os

# 保存路径
image_filename = '/home/hyx/test/BCML/exp1(1).png'
os.makedirs(os.path.dirname(image_filename), exist_ok=True)

# OURs 平移误差和旋转误差（按列：sequence, random1, random2, random3）
trans_errors = np.array([
    [0.07, 0.87, 0.83, 0.88],
    [0.08, 0.84, 0.84, 0.87],
    [0.07, 0.85, 0.83, 0.88],
    [0.09, 0.86, 0.84, 0.89],
    [0.08, 0.93, 0.83, 0.87],
])

rot_errors = np.array([
    [4.83, 44.04, 33.18, 41.56],
    [4.60, 40.42, 37.77, 39.92],
    [5.15, 41.22, 38.28, 40.02],
    [4.94, 43.09, 39.03, 41.96],
    [4.42, 49.25, 37.18, 41.12],
])

# 标签
tasks = ['sequence', 'random1', 'random2', 'random3']
x = np.arange(len(tasks))

# 计算均值与标准差
trans_mean = trans_errors.mean(axis=0)
trans_std = trans_errors.std(axis=0)
rot_mean = rot_errors.mean(axis=0)
rot_std = rot_errors.std(axis=0)

# 颜色设计：突出 sequence
colors = ['dodgerblue', 'lightgray', 'lightgray', 'lightgray']

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ---------- 平移误差图 ----------
bars0 = axs[0].bar(x, trans_mean, yerr=trans_std, capsize=5, width=0.4, color=colors, edgecolor='black')
bars0[0].set_edgecolor('red'); bars0[0].set_linewidth(2)

axs[0].set_title('Translation Error (Lower is Better)', fontsize=14)
axs[0].set_ylabel('Error (m)', fontsize=12)
axs[0].set_xticks(x)
axs[0].set_xticklabels(tasks, fontsize=12)
axs[0].set_ylim(0, 1.4)  # 增加 y 轴上限

for i in range(len(x)):
    y = trans_mean[i] + trans_std[i] + 0.02
    axs[0].text(x[i], y, f'{trans_mean[i]:.2f}', ha='center', fontsize=12)


# ---------- 旋转误差图 ----------
bars1 = axs[1].bar(x, rot_mean, yerr=rot_std, capsize=5, width=0.4, color=colors, edgecolor='black')
bars1[0].set_edgecolor('red'); bars1[0].set_linewidth(2)

axs[1].set_title('Rotation Error (Lower is Better)', fontsize=14)
axs[1].set_ylabel('Error (°)', fontsize=12)
axs[1].set_xticks(x)
axs[1].set_xticklabels(tasks, fontsize=12)
axs[1].set_ylim(0, 80)

for i in range(len(x)):
    y = rot_mean[i] + rot_std[i] + 0.8
    axs[1].text(x[i], y, f'{rot_mean[i]:.1f}', ha='center', fontsize=12)

# 调整布局并保存
plt.tight_layout()
fig.savefig(image_filename, dpi=300)
plt.show()
