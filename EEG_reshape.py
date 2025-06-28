import scipy.io as sio  
import numpy as np 
import os

def EEG_reshape(file_path):  
    data = sio.loadmat(file_path)  
    field_name = os.path.splitext(os.path.basename(file_path))[0]
    # 假设'data1_reshaped'是包含三维数组的键  
    if field_name in data:
        x = data[field_name]
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'sub01_seq500' in data:  
        x = data['sub01_seq500']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data      
    if 'sub01_seq100' in data:  
        x = data['sub01_seq100']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data      
    
    if 'sub01_seq200' in data:  
        x = data['sub01_seq200']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data      
    if 'sub01_seq50' in data:  
        x = data['sub01_seq50']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'sub01_seq300' in data:  
        x = data['sub01_seq300']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    
    if 'qjyori_1' in data:  
        x = data['qjyori_1']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'qjyori_2' in data:  
        x = data['qjyori_2']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'qjyori_3' in data:  
        x = data['qjyori_3']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'day1' in data:  
        x = data['day1']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'night_ori' in data:  
        x = data['night_ori']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    
    
    
    
    
    
    
    if 'data1_reshaped' in data:  
        x = data['data1_reshaped']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'data2_reshaped' in data:  
        x = data['data2_reshaped']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'data3_reshaped' in data:  
        x = data['data3_reshaped']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    
    if 'data4_reshaped' in data:  
        x = data['data4_reshaped']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    
    if 'data5_reshaped' in data:  
        x = data['data5_reshaped']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'Data1' in data:  
        x = data['Data1' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'Data11' in data:  
        x = data['Data11' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'Data2' in data:  
        x = data['Data2' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'Data3' in data:  
        x = data['Data3' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'Data4' in data:  
        x = data['Data4' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'Data5' in data:  
        x = data['Data5' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'Data6' in data:  
        x = data['Data6' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'Data7' in data:  
        x = data['Data7' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data7_reshaped' in data:  
        x = data['data7_reshaped']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'data6_reshaped' in data:  
        x = data['data6_reshaped']  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))  
        return reshaped_data  
    if 'data1' in data:  
        x = data['data1' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data6' in data:  
        x = data['data6' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'DATA6' in data:  
        x = data['DATA6' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'DATA1' in data:  
        x = data['DATA1' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data2' in data:  
        x = data['data2' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data3' in data:  
        x = data['data3' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data4' in data:  
        x = data['data4' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data5' in data:  
        x = data['data5' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data6' in data:  
        x = data['data6' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data7' in data:  
        x = data['data7' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data8' in data:  
        x = data['data8' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data9' in data:  
        x = data['data9' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'data10' in data:  
        x = data['data10' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'D5' in data:  
        x = data['D5' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'D1' in data:  
        x = data['D1' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'D2' in data:  
        x = data['D2' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'D3' in data:  
        x = data['D3' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'D4' in data:  
        x = data['D4' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'D6' in data:  
        x = data['D1' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'D7' in data:  
        x = data['D2' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'D8' in data:  
        x = data['D3' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'da1' in data:  
        x = data['da1' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'da2' in data:  
        x = data['da2' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'da3' in data:  
        x = data['da3' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'd1' in data:  
        x = data['d1' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'd2' in data:  
        x = data['d2' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'd3' in data:  
        x = data['d3' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'd4' in data:  
        x = data['d4' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'd5' in data:  
        x = data['d5' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'd6' in data:  
        x = data['d6' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'da4' in data:  
        x = data['da4' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'da5' in data:  
        x = data['da5' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  
    if 'da6' in data:  
        x = data['da6' ]  
        # 转置数组，从(60, 500, 180)到(180, 60, 500)  
        reshaped_data = np.transpose(x, axes=(2, 0, 1))
        return reshaped_data  



def quaternion_to_rotation_matrix(q):  
    """  
    Convert quaternion coefficients (in w, x, y, z order) to a rotation matrix.  
    """  
    w, x, y, z = q  
    Nq = np.dot(q, q)  
    s = 2.0 / Nq  
    X = x * s  
    Y = y * s  
    Z = z * s  
    wX = w * X  
    wY = w * Y  
    wZ = w * Z  
    xX = x * X  
    xY = x * Y  
    xZ = x * Z  
    yY = y * Y  
    yZ = y * Z  
    zZ = z * Z  
  
    rotation_matrix = np.array([  
        [1 - (yY + zZ), xY - wZ, xZ + wY],  
        [xY + wZ, 1 - (xX + zZ), yZ - wX],  
        [xZ - wY, yZ + wX, 1 - (xX + yY)]  
    ], dtype=np.float64)  
  
    return rotation_matrix  
  
def find_closest_timestamp(timestamps, target_timestamp):  
    min_diff = float('inf')  
    closest_idx = None  
    for i, timestamp in enumerate(timestamps):  
        diff = abs(timestamp - target_timestamp)  
        if diff < min_diff:  
            min_diff = diff  
            closest_idx = i  
    return closest_idx  
  
def read_groundtruth(filename0, filename1,EEG=True):  
    poses = []  
    timestamp0, timestamp1 = [], []  
  
    # 读取两个文件的时间戳  
    with open(filename0, 'r') as file0:  
        next(file0)  # 跳过头部  
        for line in file0:  
            if not line.startswith("#"):  
                data = line.split()  
                timestamp0.append(float(data[0]))  
  
    with open(filename1, 'r') as file1:  
        next(file1)  # 跳过头部  
        for line in file1:  
            if not line.startswith("#"):  
                data = line.split()  
                timestamp1.append(float(data[0]))  
  
    # 对于filename1中的每个时间戳，找到filename0中最接近的pose  
    with open(filename0, 'r') as file0:  
        next(file0)  # 跳过头部  
        next(file0)  # 跳过头部  
        next(file0)  # 跳过头部  
        lines0 = file0.readlines()  # 读取所有行到列表中，以便通过索引访问  
  
    for i, t1 in enumerate(timestamp1):  
        closest_idx = find_closest_timestamp(timestamp0, t1)  
        if closest_idx is not None:  
            line = lines0[closest_idx]  
            data = line.split()  
            tx, ty, tz = float(data[1]), float(data[2]), float(data[3])  
            qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])  
            T=np.array([tx, ty, tz, qw, qx, qy, qz])  
            poses.append(T)  
    return poses  



#grouthtruth_txt='/root/autodl-tmp/rgbd_dataset_freiburg1_desk/groundtruth.txt'
#rgb_txt="/root/autodl-tmp/rgbd_dataset_freiburg1_desk/rgb.txt"
#poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)  
#poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)