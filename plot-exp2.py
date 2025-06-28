import numpy as np
import matplotlib.pyplot as plt
import os

# 保存路径
image_filename = '/home/hyx/test/BCML/exp2.png'
os.makedirs(os.path.dirname(image_filename), exist_ok=True)

# OURs 平移误差和旋转误差（列为 T=50ms~500ms）
trans_errors = np.array([
    [0.10, 0.07, 0.11, 0.17, 0.17],
    [0.10, 0.08, 0.10, 0.14, 0.41],
    [0.08, 0.07, 0.10, 0.20, 0.12],
    [0.07, 0.09, 0.10, 0.30, 0.20],
    [0.10, 0.08, 0.10, 0.18, 0.12],
])

rot_errors = np.array([
    [6.26, 4.83, 6.62, 9.96, 9.60],
    [5.38, 4.60, 5.83, 8.41, 20.55],
    [4.94, 5.15, 5.66,10.19, 7.83],
    [5.15, 4.94, 5.40,14.62,11.34],
    [5.93, 4.42, 5.79, 9.96, 6.50],
])

# 时间窗口标签
tasks = ['T50', 'T100', 'T200', 'T300', 'T500']
x = np.arange(len(tasks))

# 均值与标准差
trans_mean = trans_errors.mean(axis=0)
trans_std = trans_errors.std(axis=0)
rot_mean = rot_errors.mean(axis=0)
rot_std = rot_errors.std(axis=0)

# 突出短时间窗口（T50、T100）颜色
colors = ['steelblue', 'dodgerblue', 'lightgray', 'lightgray', 'lightgray']

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ---------- 平移误差 ----------
bars0 = axs[0].bar(x, trans_mean, yerr=trans_std, capsize=5, width=0.5,color=colors, edgecolor='black')
# 强调 T50、T100
bars0[0].set_edgecolor('red'); bars0[0].set_linewidth(2)
bars0[1].set_edgecolor('red'); bars0[1].set_linewidth(2)

axs[0].set_title('Translation Error(Lower is Better)', fontsize=14)
axs[0].set_ylabel('Error (m)', fontsize=12)
axs[0].set_xticks(x)
axs[0].set_xticklabels(tasks, fontsize=12)
axs[0].set_ylim(0, max(trans_mean + trans_std) + 0.2)

# 数值标签
for i in range(len(x)):
    y = trans_mean[i] + trans_std[i] + 0.01
    axs[0].text(x[i], y, f'{trans_mean[i]:.2f}', ha='center', fontsize=12)

# ---------- 旋转误差 ----------
bars1 = axs[1].bar(x, rot_mean, yerr=rot_std, capsize=5, width=0.5,color=colors, edgecolor='black')
# 强调 T50、T100
bars1[0].set_edgecolor('red'); bars1[0].set_linewidth(2)
bars1[1].set_edgecolor('red'); bars1[1].set_linewidth(2)

axs[1].set_title('Rotation Error(Lower is Better)', fontsize=14)
axs[1].set_ylabel('Error (°)', fontsize=12)
axs[1].set_xticks(x)
axs[1].set_xticklabels(tasks, fontsize=12)
axs[1].set_ylim(0, max(rot_mean + rot_std) + 5)

for i in range(len(x)):
    y = rot_mean[i] + rot_std[i] + 0.2
    axs[1].text(x[i], y, f'{rot_mean[i]:.1f}', ha='center', fontsize=12)
plt.tight_layout()
fig.savefig(image_filename, dpi=300)
plt.show()
