import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import hilbert
from scipy.interpolate import UnivariateSpline
import scipy.io as sio
from scipy.signal import find_peaks
def find_peak_after_time(peaks, time_point):
    # 找第一个峰的位置 > time_point
    for p in peaks:
        if p >= time_point:
            return p
    return None
import numpy as np
def gaussian(x, A, mu, sigma, offset):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset
def smooth(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
mat_paths = [
    "/home/hyx/实验五/Subject01/ica/sub01_seq_100.mat",
    "/home/hyx/实验五/Subject02/ica/sub02_seq_100.mat",
    "/home/hyx/实验五/Subject03/ica/sub03_seq_100.mat",
    "/home/hyx/实验五/Subject04/ica/sub04_seq_100.mat",
    "/home/hyx/实验五/Subject05/ica/sub05_seq_100.mat"
]

var_names = [f"sub{idx+1:02d}_seq_100" for idx in range(5)]

# 分别读取并赋值为 sub1_data, sub2_data, ...
for i in range(5):
    mat = sio.loadmat(mat_paths[i])
    globals()[f"sub{i+1}_data"] = mat[var_names[i]]

chanlocs_file = "/home/hyx/60locations.locs"
chanlocs = []
with open(chanlocs_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            chanlocs.append(parts[-1])  # 取每行最后一个元素，作为通道名

print("通道标签示例：", chanlocs[:10])
window_size = 6
all_subs_data = [sub1_data, sub2_data, sub3_data, sub4_data, sub5_data]

n_channels = 60  # 通道数，修改为你的真实通道数
save_dir = "/home/hyx/test/BCML/exp6"
os.makedirs(save_dir, exist_ok=True)
erp_data=[]
sub_window_means = []
for c in range(60):
    sub_window_means = []
    for sub_idx, sub_data in enumerate(all_subs_data):
        n_channels, n_times, n_frames = sub_data.shape
        windows_per_sub = n_frames - window_size + 1
        step = window_size
        windows_per_sub = (n_frames - window_size) // step + 1
        windows = np.zeros((windows_per_sub, n_times, window_size))

        # for w in range(windows_per_sub):
        #     windows[w, :, :] = sub_data[c, :, w:w + window_size]
        for i, w in enumerate(range(0, n_frames - window_size + 1, step)):
            windows[i, :, :] = sub_data[c, :, w:w + window_size]
        avg_windows = np.mean(windows, axis=0)

        sub_window_means.append(avg_windows)
    all_subs_array = np.stack(sub_window_means, axis=0)
    erp_avg = np.mean(all_subs_array, axis=0)            # shape: (n_times, window_size)

    erp_avg_reshaped = erp_avg.reshape(-1, 1, order='F')  # 按列展开
    erp_data.append(erp_avg_reshaped) 
    plt.figure(figsize=(8, 4))
    y = erp_avg_reshaped.flatten()  


    y_smooth = savgol_filter(y, window_length=30, polyorder=2)
    x = np.arange(len(y))  
    y_corrected = y_smooth - y_smooth[0]


    valleys, _ = find_peaks(-y_corrected)

    def find_nth_valley_after_time(valleys, time_point, n=2):
        # 找所有在time_point之后的波谷
        valleys_after = [v for v in valleys if v >= time_point]
        if len(valleys_after) >= n:
            return valleys_after[n-1]
        else:
            return None

    valley_100ms_2nd = find_nth_valley_after_time(valleys, 100, n=2)
    valley_200ms_2nd = find_nth_valley_after_time(valleys, 200, n=2)
    # 绘图
    plt.plot(x, y_corrected, linestyle='-')
    plt.gca().invert_yaxis() 
    plt.title(rf"Channel $\bf{{{chanlocs[c]}}}$ ERP", fontsize=25) 
    plt.xlabel('Time (ms)',fontsize=15)
    plt.ylabel('Amplitude (μV)',fontsize=15)
    # if valley_100ms_2nd is not None:
    #     plt.axvline(x=valley_100ms_2nd, color='red', linestyle='--', label='2nd Valley after 100ms')
    #     #plt.text(valley_100ms_2nd, plt.ylim()[1]*0.9, '2nd Valley after 100ms', rotation=90, color='red', fontsize=12, va='top')

    # if valley_200ms_2nd is not None:
    #     plt.axvline(x=valley_200ms_2nd, color='blue', linestyle='--', label='2nd Valley after 200ms')
    #     #plt.text(valley_200ms_2nd, plt.ylim()[1]*0.9, '2nd Valley after 200ms', rotation=90, color='blue', fontsize=12, va='top')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ERP_channel_{chanlocs[c]}.png"))
    plt.close()

print(f"所有通道ERP图已保存至目录：{save_dir}")