# Imports
import os
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy import signal
from PIL import Image
import pandas as pd
import torch.utils.data as Data
import numpy as np
import scipy.io as sci
import os.path as osp
from EEG_reshape import EEG_reshape,read_groundtruth,quaternion_to_rotation_matrix,find_closest_timestamp
import cv2
from read_pose import read_pose
import scipy.io as sio 
import re 
from utils_ import  quaternion_angular_error,process_poses,load_state_dict,load_image,qlog
from interpolate_poses import interpolate_ins_poses
import random

def image_load(image_path, image_name):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_file = os.path.join(image_path, image_name)
    image = Image.open(image_file).convert('RGB')
    image = transform(image)
    return image

def image_loader(image_path):
    image_dict = {}
    for image_name in os.listdir(image_path):
        image = image_load(image_path, image_name)
        image_dict[image_name[:-5]] = image
    return image_dict

# Dataset class
class EEGDataset:
    # Constructor
    def __init__(self, opt):
        # Load EEG signals
        self.opt = opt
        loaded = torch.load(self.opt.eeg_dataset)
        self.image_dict = image_loader(opt.image_dataset)
        if opt.subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        else:
            self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.opt.time_low:self.opt.time_high,:]
        if self.opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1,128,self.opt.time_high-self.opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Get image
        image_name = self.images[self.data[i]["image"]]
        image = self.image_dict[image_name]
        # Return
        return eeg, label, image

# Splitter class
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data                                                                                   
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        HZ = 1000
        low_f, high_f = 5, 95
        b, a = signal.butter(2, [low_f*2/HZ, high_f*2/HZ], 'bandpass')
        for i in self.split_idx:
            self.dataset.data[i]["eeg"] = torch.from_numpy(signal.lfilter(b, a, self.dataset.data[i]["eeg"]).copy())
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label, image = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label, image

def data_loader3(opt):      

    sample_size = 300

    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size / 3)]
    test_indices = indices[-int(sample_size / 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 100
    channels = 1



    EEG_path="/home/hyx/subject3//indoor2.mat"


    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk2/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk2/rgb.txt"


    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    
    

    from scipy.spatial.transform import Rotation as R

    save_dir = '/home/hyx/test/BCML/memory_trail'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'desk2.txt')


    eeg_poses = np.array(eeg_poses[:300])
    positions = eeg_poses[:, :3]
    trans_deltas = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    total_trans = np.sum(trans_deltas)

    quaternions = eeg_poses[:, 3:]
    rot1 = R.from_quat(quaternions[1:])
    rot0 = R.from_quat(quaternions[:-1])
    relative_rot = rot1 * rot0.inv()
    rot_deltas = np.rad2deg(relative_rot.magnitude())
    total_rot = np.sum(rot_deltas)

    # 输出文本
    with open(save_path, 'w') as f:
        f.write(f"平均位移变化: {np.mean(trans_deltas):.4f} m\n")
        f.write(f"总位移距离: {total_trans:.2f} m\n")
        f.write(f"平均旋转变化: {np.mean(rot_deltas):.2f}°\n")
        f.write(f"总角度变化: {total_rot:.2f}°\n")

    print(f"结果已保存到：{save_path}")

    
    
    
    
    
    
    
    
    
    
    
    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(sample_size)
    #frame_idx=numbers  

    c_imgs = [osp.join('/home/hyx/test/BCML/data/Work/desk2', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if opt.mode == 'train':
        return_dataloader = train_dataloader
    elif opt.mode == 'val':
        return_dataloader = test_dataloader
    else:
        return_dataloader = test_dataloader

    return train_dataloader, test_dataloader

def data_loader5(opt):    #KKTTI
    BATCH_SIZE = opt.batch_size                       
    sample_size = 2760


    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size / 3)]
    test_indices = indices[-int(sample_size / 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')

    n_step = 60
    n_input = 100
    channels = 1

    # EEG_path="/home/hyx/subject3/实验2/ica/outdoor1.mat"
    # EEG_path1="/home/hyx/subject3/实验2/ica/outdoor2.mat"
    # EEG_path2="/home/hyx/subject3/实验2/ica/outdoor3.mat"
    # EEG_path="/home/hyx/subject3/实验2/简单带通滤波/outdoor1.mat"
    # EEG_path1="/home/hyx/subject3/实验2/简单带通滤波/outdoor2.mat"
    # EEG_path2="/home/hyx/subject3/实验2/简单带通滤波/outdoor3.mat"
    EEG_path="/home/hyx/subject3/实验2/原始数据/outdoor1.mat"
    EEG_path1="/home/hyx/subject3/实验2/原始数据/outdoor2.mat"
    EEG_path2="/home/hyx/subject3/实验2/原始数据/outdoor3.mat"

    # EEG_path="/home/hyx/subject4/实验2/ica/outdoor1.mat"
    # EEG_path1="/home/hyx/subject4/实验2/ica/outdoor2.mat"
    # EEG_path2="/home/hyx/subject4/实验2/ica/outdoor3.mat"
    # EEG_path="/home/hyx/subject4/实验2/简单带通滤波/outdoor1.mat"
    # EEG_path1="/home/hyx/subject4/实验2/简单带通滤波/outdoor2.mat"
    # EEG_path2="/home/hyx/subject4/实验2/简单带通滤波/outdoor3.mat"
    # EEG_path="/home/hyx/subject4/实验2/原始数据/outdoor1.mat"
    # EEG_path1="/home/hyx/subject4/实验2/原始数据/outdoor2.mat"
    # EEG_path2="/home/hyx/subject4/实验2/原始数据/outdoor3.mat"
    
    
    # EEG_path="/home/hyx/subject5/实验2/outdoor1.mat"
    # EEG_path1="/home/hyx/subject5/实验2/outdoor2.mat"
    # EEG_path2="/home/hyx/subject5/实验2/outdoor3.mat"
    # EEG_path="/home/hyx/subject4/实验2/简单带通滤波/outdoor1.mat"
    # EEG_path1="/home/hyx/subject4/实验2/简单带通滤波/outdoor2.mat"
    # EEG_path2="/home/hyx/subject4/实验2/简单带通滤波/outdoor3.mat"
    # EEG_path="/home/hyx/subject5/zqh原始数据/实验2/outdoor1.mat"
    # EEG_path1="/home/hyx/subject5/zqh原始数据/实验2/outdoor2.mat"
    # EEG_path2="/home/hyx/subject5/zqh原始数据/实验2/outdoor3.mat"


    # EEG_path="/home/hyx/subject2/实验2/outdoor1.mat"
    # EEG_path1="/home/hyx/subject2/实验2/outdoor2.mat"
    # EEG_path2="/home/hyx/subject2/实验2/outdoor3.mat"

    # print(sio.loadmat(EEG_path))  

    grouthtruth_txt1='/home/hyx/adalashiwork/Kiiti/data_odometry_poses/sequences/05/tum_05_gt.txt'

    eeg_data0 = EEG_reshape(EEG_path)
    eeg_data1 = EEG_reshape(EEG_path1)
    eeg_data2 = EEG_reshape(EEG_path2)
    eeg_data=np.concatenate((eeg_data0,eeg_data1,eeg_data2))
    eeg_poses = read_pose(grouthtruth_txt1)
    from scipy.spatial.transform import Rotation as R

    save_dir = '/home/hyx/test/BCML/memory_trail'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'kitti.txt')

    # 计算
    eeg_poses = np.array(eeg_poses)
    positions = eeg_poses[:, :3]
    trans_deltas = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    total_trans = np.sum(trans_deltas)

    quaternions = eeg_poses[:, 3:]
    rot1 = R.from_quat(quaternions[1:])
    rot0 = R.from_quat(quaternions[:-1])
    relative_rot = rot1 * rot0.inv()
    rot_deltas = np.rad2deg(relative_rot.magnitude())
    total_rot = np.sum(rot_deltas)

    # 输出文本
    with open(save_path, 'w') as f:
        f.write(f"平均位移变化: {np.mean(trans_deltas):.4f} m\n")
        f.write(f"总位移距离: {total_trans:.2f} m\n")
        f.write(f"平均旋转变化: {np.mean(rot_deltas):.2f}°\n")
        f.write(f"总角度变化: {total_rot:.2f}°\n")

    print(f"结果已保存到：{save_path}")

    eeg_y_test=[]
    eeg_y_train=[]

    eeg_x_test = eeg_data[test_indices]
    #eeg_x_test = eeg_data[-60:,:,:]

    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]
    #eeg_x_train = eeg_data[:120,:,:]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    #data_in = pd.read_csv(opt.facial_image_dataset + 'image_1_32.csv', header=None)
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(2760)
    #frame_idx=numbers  

    c_imgs = [osp.join('/home/hyx/adalashiwork/AtLoc-master/data/Kitti', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_pose(grouthtruth_txt1)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])





    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    #eeg_y_train = eeg_y_train.astype('float32')
    #eeg_y_test = eeg_y_test.astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0
    #image_y_train = image_y_train.astype('float32')
    #image_y_test = image_y_test.astype('float32')

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(np.array(image_y_train)), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(np.array(image_y_test)), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if opt.mode == 'train':
        return_dataloader = train_dataloader
    elif opt.mode == 'val':
        return_dataloader = test_dataloader
    else:
        return_dataloader = test_dataloader

    return train_dataloader, test_dataloader

def data_loader55(opt):
    n_step = 60
    n_input = 100
    channels = 1
    
    # 使用相同的测试序列
    #["/2014-12-09-13-21-02"]
    seqs = ["/2014-12-09-13-21-02"]
    
    image_x, image_y = [], []
    ps, ts = {}, {}
    vo_stats = {}

    # 读取并处理数据
    for seq in seqs:
        seq_dir = osp.join("/mnt/HYX", seq.strip("/"))
        
        # 读取时间戳
        ts_filename = osp.join(seq_dir, 'stereo.timestamps')
        with open(ts_filename, 'r') as f:
            ts[seq] = [int(l.rstrip().split(' ')[0]) for i, l in enumerate(f) if 318 < i < 4519]  # 过滤时间戳
        
        # 读取位姿数据并插值
        pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
        p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq]))
        ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
        vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

    # 处理位姿数据
    poses = np.empty((0, 12))
    for p in ps.values():
        poses = np.vstack((poses, p))

    # 计算均值和标准差
    mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)
    std_t = np.std(poses[:, [3, 7, 11]], axis=0)

    # 处理位姿数据
    poses = np.empty((0, 6))
    for seq in seqs:
        pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                            align_s=vo_stats[seq]['s'])
        poses = np.vstack((poses, pss))
    from scipy.spatial.transform import Rotation as R

    save_dir = '/home/hyx/test/BCML/memory_trail'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rotation_translation_stats.txt')

    positions = poses[:, :3]
    trans_deltas = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    total_trans = np.sum(trans_deltas)
    mean_trans = np.mean(trans_deltas)

    rotvecs = poses[:, 3:]
    rotations = R.from_rotvec(rotvecs)
    rot1 = rotations[1:]
    rot0 = rotations[:-1]
    relative_rot = rot1 * rot0.inv()
    rot_deltas = np.rad2deg(relative_rot.magnitude())
    total_rot = np.sum(rot_deltas)
    mean_rot = np.mean(rot_deltas)

    # 读取归一化参数
    pose_stats_file = osp.join('/home/hyx/adalashiwork/AtLoc-master/data/RobotCar/full/pose_stats.txt')
    pose_stats = np.loadtxt(pose_stats_file)  # shape (2, 7)
    pose_m = pose_stats[0]
    pose_s = pose_stats[1]

    # 取位移维度均值和标准差（前三维）
    pose_m_trans = np.mean(pose_m[:3])
    pose_s_trans = np.mean(pose_s[:3])

    # 反归一化位移误差
    mean_trans_denorm = mean_trans * pose_s_trans + pose_m_trans
    total_trans_denorm = total_trans * pose_s_trans + pose_m_trans

    # 保存结果
    with open(save_path, 'w') as f:
        f.write(f"平均位移变化 (反归一化): {mean_trans_denorm:.4f} m\n")
        f.write(f"总位移距离 (反归一化): {total_trans_denorm:.2f} m\n")
        f.write(f"平均旋转变化 (未反归一化): {mean_rot:.2f}°\n")
        f.write(f"总角度变化 (未反归一化): {total_rot:.2f}°\n")

    print(f"结果已保存到：{save_path}")

    print("image_loader_ok")
    # c_imgs = [osp.join(seq_dir, 'rgb', '{:d}.png'.format(int(i))) for i in range(320,4520)]
    c_imgs = [osp.join(seq_dir, 'rgb', '{:d}.png'.format(int(i))) for i in range(320,4520)]
    for img_path, pose in zip(c_imgs, poses):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue  
        img = cv2.resize(img, (224, 224))  
        img_new = np.expand_dims(img, axis=-1)  # 增加最后一个维度
        img_flat = img_new.flatten()  # 拉平成一维
        image_x.append(img_flat)  # 存储图像数据
        image_y.append(pose)  # 存储位姿数据

    # 加载EEG数据
    EEG_test_path = "/home/hyx/实验4/Day/实验4_new/原始数据/day.mat"
    eeg_x = EEG_reshape(EEG_test_path)

    # 随机打乱索引
    indices = np.arange(len(image_x))
    np.random.shuffle(indices)

    # 划分训练集和测试集
    split_idx = int(0.7 * len(image_x))  
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    image_x_train = [image_x[i] for i in train_indices]
    image_y_train = [image_y[i] for i in train_indices]
    image_x_test = [image_x[i] for i in test_indices]
    image_y_test = [image_y[i] for i in test_indices]
    eeg_x_train=eeg_x[train_indices]
    eeg_x_test = eeg_x[test_indices]  # 对应EEG数据的测试集部分

    print("image_loader_ok")

    height = 224
    width = 224
    channels = 3

    # 转换数据形状
    image_x_train = np.array(image_x_train)
    image_x_test = np.array(image_x_test)

    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')

    image_x_train = image_x_train.transpose(0, 3, 1, 2)  # 转换成 (batch_size, channels, height, width)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)

    # 归一化
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_test = np.array(eeg_x_test)
    image_y_train = np.array(image_y_train)
    image_y_test = np.array(image_y_test)

    # 创建TensorDataset和DataLoader
    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)

    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if opt.mode == 'train':
        return_dataloader = train_dataloader
    elif opt.mode == 'val':
        return_dataloader = test_dataloader
    else:
        return_dataloader = test_dataloader

    return train_dataloader, test_dataloader
def data_loader555(opt):
    n_step = 60
    n_input = 100
    channels = 1
    
    # 使用相同的测试序列
    #["/2014-12-09-13-21-02"]
    seqs = ["/2014-12-16-18-44-24"]
    
    image_x, image_y = [], []
    ps, ts = {}, {}
    vo_stats = {}

    # 读取并处理数据
    for seq in seqs:
        seq_dir = osp.join("/mnt/HYX", seq.strip("/"))
        
        # 读取时间戳
        ts_filename = osp.join(seq_dir, 'stereo.timestamps')
        with open(ts_filename, 'r') as f:
            ts[seq] = [int(l.rstrip().split(' ')[0]) for i, l in enumerate(f) if 1898 < i < 5599]  # 过滤时间戳
        
        # 读取位姿数据并插值
        pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
        p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq]))
        ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
        vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

    # 处理位姿数据
    poses = np.empty((0, 12))
    for p in ps.values():
        poses = np.vstack((poses, p))

    # 计算均值和标准差
    mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)
    std_t = np.std(poses[:, [3, 7, 11]], axis=0)

    # 处理位姿数据
    poses = np.empty((0, 6))
    for seq in seqs:
        pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                            align_s=vo_stats[seq]['s'])
        poses = np.vstack((poses, pss))

    # 读取图片数据
    print("image_loader_ok")
    EEG_test_path = "/home/hyx/实验4/Day/实验4_new/原始数据/night.mat"
    print(sio.loadmat(EEG_test_path))
    eeg_x = EEG_reshape(EEG_test_path)

    # c_imgs = [osp.join(seq_dir, 'rgb', '{:d}.png'.format(int(i))) for i in range(320,4520)]
    c_imgs = [osp.join(seq_dir, 'rgb', '{:d}.png'.format(int(i))) for i in range(1900,5600)]
    for img_path, pose in zip(c_imgs, poses):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue  
        img = cv2.resize(img, (224, 224))  
        img_new = np.expand_dims(img, axis=-1)  # 增加最后一个维度
        img_flat = img_new.flatten()  # 拉平成一维
        image_x.append(img_flat)  # 存储图像数据
        image_y.append(pose)  # 存储位姿数据

    # 加载EEG数据

    # 随机打乱索引
    indices = np.arange(len(image_x))
    np.random.shuffle(indices)

    # 划分训练集和测试集
    split_idx = int(0.7 * len(image_x))  
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    image_x_train = [image_x[i] for i in train_indices]
    image_y_train = [image_y[i] for i in train_indices]
    image_x_test = [image_x[i] for i in test_indices]
    image_y_test = [image_y[i] for i in test_indices]
    eeg_x_train=eeg_x[train_indices]
    eeg_x_test = eeg_x[test_indices]  # 对应EEG数据的测试集部分

    print("image_loader_ok")

    height = 224
    width = 224
    channels = 3

    # 转换数据形状
    image_x_train = np.array(image_x_train)
    image_x_test = np.array(image_x_test)

    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')

    image_x_train = image_x_train.transpose(0, 3, 1, 2)  # 转换成 (batch_size, channels, height, width)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)

    # 归一化
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_test = np.array(eeg_x_test)
    image_y_train = np.array(image_y_train)
    image_y_test = np.array(image_y_test)

    # 创建TensorDataset和DataLoader
    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)

    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if opt.mode == 'train':
        return_dataloader = train_dataloader
    elif opt.mode == 'val':
        return_dataloader = test_dataloader
    else:
        return_dataloader = test_dataloader

    return train_dataloader, test_dataloader

def data_loader6(opt):      # 脑电数据时长500 desk1 数据 
                         
    sample_size = 500
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 500
    channels = 1

    print(train_indices)

    EEG_path="/home/hyx/实验五/Subject01/NP/sub01_seq_500.mat"
    #"/home/hyx/subject2/实验1/order1.mat"
    
    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(500)
    #frame_idx=numbers  

    c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb', 'test1({:d}).png'.format(int(i+1)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    return train_dataloader,test_dataloader


def data_loader7(opt):      # 脑电数据时长100 desk1 数据 
                         
    sample_size = 500
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 100
    channels = 1

    print(train_indices)

    EEG_path="/home/hyx/实验五/Subject01/NP/sub01_seq_100.mat"
    #"/home/hyx/subject2/实验1/order1.mat"
    
    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
        
    from scipy.spatial.transform import Rotation as R

    save_dir = '/home/hyx/test/BCML/memory_trail'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'desk1.txt')

    # 计算
    eeg_poses = np.array(eeg_poses[:500])
    positions = eeg_poses[:, :3]
    trans_deltas = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    total_trans = np.sum(trans_deltas)

    quaternions = eeg_poses[:, 3:]
    rot1 = R.from_quat(quaternions[1:])
    rot0 = R.from_quat(quaternions[:-1])
    relative_rot = rot1 * rot0.inv()
    rot_deltas = np.rad2deg(relative_rot.magnitude())
    total_rot = np.sum(rot_deltas)

    # 输出文本
    with open(save_path, 'w') as f:
        f.write(f"平均位移变化: {np.mean(trans_deltas):.4f} m\n")
        f.write(f"总位移距离: {total_trans:.2f} m\n")
        f.write(f"平均旋转变化: {np.mean(rot_deltas):.2f}°\n")
        f.write(f"总角度变化: {total_rot:.2f}°\n")

    print(f"结果已保存到：{save_path}")


    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(500)
    #frame_idx=numbers  

    c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb', 'test1({:d}).png'.format(int(i+1)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    return train_dataloader,test_dataloader

def data_loader8(opt):      # 脑电数据时长200 desk1 数据 
                         
    sample_size = 500
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 200
    channels = 1

    print(train_indices)

    EEG_path="/home/hyx/实验五/Subject01/NP/sub01_seq_200.mat"
    #"/home/hyx/subject2/实验1/order1.mat"
    
    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(500)
    #frame_idx=numbers  

    c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb', 'test1({:d}).png'.format(int(i+1)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    return train_dataloader,test_dataloader

def data_loader9(opt):      # 脑电数据时长50 desk1 数据 
                         
    sample_size = 500
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 50
    channels = 1

    print(train_indices)

    EEG_path="/home/hyx/实验五/Subject01/NP/sub01_seq_50.mat"
    #"/home/hyx/subject2/实验1/order1.mat"
    
    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(500)
    #frame_idx=numbers  

    c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb', 'test1({:d}).png'.format(int(i+1)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    return train_dataloader,test_dataloader

def data_loader10(opt):      # 脑电数据时长300 desk1 数据 
                         
    sample_size = 500
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 300
    channels = 1

    print(train_indices)

    EEG_path="/home/hyx/实验五/Subject05/NP/sub05_seq_300.mat"
    #"/home/hyx/subject2/实验1/order1.mat"
    
    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(500)
    #frame_idx=numbers  

    c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb', 'test1({:d}).png'.format(int(i+1)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    return train_dataloader,test_dataloader

def data_loader11(opt):      # 脑电数据时长100 desk1 数据 随机
                         
    sample_size = 500
    n_step = 60
    n_input = 100
    channels = 1


    EEG_path="/home/hyx/实验五/Subject01/NP/sub01_random3_100.mat"
    
    #"/home/hyx/subject1/random3.mat"
    #"/home/hyx/subject5/实验1/random3.mat"
    #"/home/hyx/subject3/实验1/ica/random1.mat"
    #"/home/hyx/subject2/实验1/random3.mat"

    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

    print(sio.loadmat(EEG_path))  
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    if "random1" in EEG_path:
        folder_path = '/mnt/HYX/rgbd_dataset_freiburg1_desk/random_1'
        pattern = re.compile(r'\((\d+)\)')
        file_names = os.listdir(folder_path)
        sorted_files = sorted(file_names, key=lambda name: int(re.findall(r'\d+', name)[0]) if re.findall(r'\d+', name) else float('inf'))

        indices = []
        for name in sorted_files:
            match = pattern.search(name)
            if match:
                indices.append(int(match.group(1)))
        train_indices = indices[:334]
        test_indices = indices[334:] 
        c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg1_desk/random_1', '{:d}_test({:d}).png'.format(i+1, idx)) for i, idx in enumerate(indices[:500])]
    
    if "random2" in EEG_path:
        folder_path = '/mnt/HYX/rgbd_dataset_freiburg1_desk/random_2'
        pattern = re.compile(r'\((\d+)\)')
        file_names = os.listdir(folder_path)
        sorted_files = sorted(file_names, key=lambda name: int(re.findall(r'\d+', name)[0]) if re.findall(r'\d+', name) else float('inf'))

        indices = []
        for name in sorted_files:
            match = pattern.search(name)
            if match:
                indices.append(int(match.group(1)))
        train_indices = indices[:334]
        test_indices = indices[334:] 
        c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg1_desk/random_2', '{:d}_test({:d}).png'.format(i+1, idx)) for i, idx in enumerate(indices[:500])]
    if "random3" in EEG_path:
        folder_path = '/mnt/HYX/rgbd_dataset_freiburg1_desk/random_3'
        pattern = re.compile(r'\((\d+)\)')
        file_names = os.listdir(folder_path)
        sorted_files = sorted(file_names, key=lambda name: int(re.findall(r'\d+', name)[0]) if re.findall(r'\d+', name) else float('inf'))

        indices = []
        for name in sorted_files:
            match = pattern.search(name)
            if match:
                indices.append(int(match.group(1)))
        train_indices = indices[:334]
        test_indices = indices[334:] 
        c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg1_desk/random_3', '{:d}_test({:d}).png'.format(i+1, idx)) for i, idx in enumerate(indices[:500])]
    
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    
    
    
    
    
    
    
    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[334:]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i-1])

    eeg_x_train = eeg_data[:334]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i-1])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)

    image_x_test=image_data[334:]
    image_x_train=image_data[:334]




    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    eeg_y_train = np.array(eeg_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    eeg_y_test = np.array(eeg_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(eeg_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(eeg_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if opt.mode == 'train':
        return_dataloader = train_dataloader
    elif opt.mode == 'val':
        return_dataloader = test_dataloader
    else:
        return_dataloader = test_dataloader

    return train_dataloader,test_dataloader

def data_loader14(opt):      # dyanamic-xyz 
                         
    sample_size = 859
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 100
    channels = 1

    print(train_indices)

    EEG_path="/home/hyx/实验4/Dynamic/新实验4/原始数据/qjy_seq01.mat"
    #"/home/hyx/subject2/实验1/order1.mat"
    
    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg3_walking_xyz/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    
    from scipy.spatial.transform import Rotation as R

    save_dir = '/home/hyx/test/BCML/memory_trail'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'xyz.txt')

    # 计算
    eeg_poses = np.array(eeg_poses)
    positions = eeg_poses[:, :3]
    trans_deltas = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    total_trans = np.sum(trans_deltas)

    quaternions = eeg_poses[:, 3:]
    rot1 = R.from_quat(quaternions[1:])
    rot0 = R.from_quat(quaternions[:-1])
    relative_rot = rot1 * rot0.inv()
    rot_deltas = np.rad2deg(relative_rot.magnitude())
    total_rot = np.sum(rot_deltas)

    # 输出文本
    with open(save_path, 'w') as f:
        f.write(f"平均位移变化: {np.mean(trans_deltas):.4f} m\n")
        f.write(f"总位移距离: {total_trans:.2f} m\n")
        f.write(f"平均旋转变化: {np.mean(rot_deltas):.2f}°\n")
        f.write(f"总角度变化: {total_rot:.2f}°\n")

    print(f"结果已保存到：{save_path}")

    
    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(859)
    #frame_idx=numbers  

    c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg3_walking_xyz/order', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    return train_dataloader,test_dataloader

def data_loader15(opt):      # dyanamic-rpy 
                         
    sample_size = 910
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 100
    channels = 1

    print(train_indices)

    EEG_path="/home/hyx/实验4/Dynamic/新实验4/原始数据/qjy_seq02.mat"
    #"/home/hyx/subject2/实验1/order1.mat"
    
    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    from scipy.spatial.transform import Rotation as R

    save_dir = '/home/hyx/test/BCML/memory_trail'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rpy.txt')

    # 计算
    eeg_poses = np.array(eeg_poses)
    positions = eeg_poses[:, :3]
    trans_deltas = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    total_trans = np.sum(trans_deltas)

    quaternions = eeg_poses[:, 3:]
    rot1 = R.from_quat(quaternions[1:])
    rot0 = R.from_quat(quaternions[:-1])
    relative_rot = rot1 * rot0.inv()
    rot_deltas = np.rad2deg(relative_rot.magnitude())
    total_rot = np.sum(rot_deltas)

    # 输出文本
    with open(save_path, 'w') as f:
        f.write(f"平均位移变化: {np.mean(trans_deltas):.4f} m\n")
        f.write(f"总位移距离: {total_trans:.2f} m\n")
        f.write(f"平均旋转变化: {np.mean(rot_deltas):.2f}°\n")
        f.write(f"总角度变化: {total_rot:.2f}°\n")

    print(f"结果已保存到：{save_path}")

    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(910)
    #frame_idx=numbers  

    c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/order', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    return train_dataloader,test_dataloader

def data_loader16(opt):      # dyanamic-sphere
                         
    sample_size = 1067
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 100
    channels = 1

    print(train_indices)

    EEG_path="/home/hyx/实验4/Dynamic/新实验4/原始数据/qjy_seq03.mat"
    #"/home/hyx/subject2/实验1/order1.mat"
    
    grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg3_walking_halfsphere/groundtruth.txt'
    rgb_txt="/mnt/HYX/rgbd_dataset_freiburg3_walking_halfsphere/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    from scipy.spatial.transform import Rotation as R

    save_dir = '/home/hyx/test/BCML/memory_trail'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'sphere.txt')

    # 计算
    eeg_poses = np.array(eeg_poses)
    positions = eeg_poses[:, :3]
    trans_deltas = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    total_trans = np.sum(trans_deltas)

    quaternions = eeg_poses[:, 3:]
    rot1 = R.from_quat(quaternions[1:])
    rot0 = R.from_quat(quaternions[:-1])
    relative_rot = rot1 * rot0.inv()
    rot_deltas = np.rad2deg(relative_rot.magnitude())
    total_rot = np.sum(rot_deltas)

    # 输出文本
    with open(save_path, 'w') as f:
        f.write(f"平均位移变化: {np.mean(trans_deltas):.4f} m\n")
        f.write(f"总位移距离: {total_trans:.2f} m\n")
        f.write(f"平均旋转变化: {np.mean(rot_deltas):.2f}°\n")
        f.write(f"总角度变化: {total_rot:.2f}°\n")

    print(f"结果已保存到：{save_path}")

    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(1067)
    #frame_idx=numbers  

    c_imgs = [osp.join('/mnt/HYX/rgbd_dataset_freiburg3_walking_halfsphere/order', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])



    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0

    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)
        

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    return train_dataloader,test_dataloader



def get_loader(opt):
    # Load DataLoader of given DialogDataset
    if opt.data =='facial' :
        train_dataloader,test_dataloader = data_loader16(opt)
    return train_dataloader,test_dataloader
