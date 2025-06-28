# Imports
import math
import numpy as np
from random import random
from config import get_config, activation_dict
from data_loader import get_loader
import torch
import numpy as np
import torch.nn as nn
from EEG_reshape import EEG_reshape,read_groundtruth
from torch.utils.data import DataLoader
import torch.utils.data as Data
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD, MMD_loss, get_shuffled,Criterion, loss_fn,qlog,apply_qlog_to_tensor,qexp,quaternion_angular_error
from utils_ import  quaternion_angular_error,process_poses,load_state_dict,load_image
from interpolate_poses import interpolate_ins_poses
from model import AverageMeter#EEGNet
import os.path as osp
from read_pose import read_pose
from matplotlib import pyplot as plt
import torch.nn.functional as F
import scipy.io as sio  
import torch.nn.functional as F
import scipy.io as sio 
import re 
import os.path as osp
import os
from torch import Tensor
from einops import rearrange
import math
from einops.layers.torch import Rearrange, Reduce
from utils_ import  quaternion_angular_error,process_poses,load_state_dict,load_image
from interpolate_poses import interpolate_ins_poses
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25
        self.C = 60
        self.T = 100
        self.weights = nn.Parameter(torch.Tensor(1, self.C, self.T))
        #self.fc_xyz = nn.Linear(feat_dim, 3)                #全连接层（nn.Linear），用于将feat_dim维的特征向量映射到3维空间
        #self.fc_wpqr = nn.Linear(feat_dim, 3)
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((30, 31, 0, 0)),
            nn.Conv2d(
                in_channels=1,          
                out_channels=8,         
                kernel_size=(1, self.C),    
                bias=False
            ),                        
            nn.BatchNorm2d(8)           
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,          
                out_channels=16,        
                kernel_size=(self.C, 1), 
                groups=8,
                bias=False
            ),                        
            nn.BatchNorm2d(16),       
            nn.ELU(),
            nn.AvgPool2d((1, 4)),    
            nn.Dropout(self.drop_out)
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
               in_channels=16,      
               out_channels=16,       
               kernel_size=(1, 16),  
               groups=16,
               bias=False
            ),                       
            nn.Conv2d(
                in_channels=16,        
                out_channels=16,    
                kernel_size=(1, 1), 
                bias=False
            ),                    
            nn.BatchNorm2d(16),         
            nn.ELU(),
            nn.AvgPool2d((1, 8)),    
            nn.Dropout(self.drop_out)
        )
        # self.linear1 = nn.Linear((16), 3)
        # self.linear2 = nn.Linear((16), 3)
        # self.linear = nn.Linear((16), 6)  
        self.linear1 = nn.Linear((self.T//100*48), 3)
        self.linear2 = nn.Linear((self.T//100*48), 3)
        self.linear = nn.Linear((self.T//100*48), 6)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(1)
        self.weights.data.uniform_(-stdv, stdv)
    def forward(self, x):
        # x = x*self.weights
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        xyz = self.linear1(x)
        wpqr=self.linear2(x)
        x=torch.cat((xyz, wpqr), 1)
        return x


if __name__ == '__main__':
    
    # Setting random seed
    random_name = str(random())
    random_seed = 42 #336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    cuda = torch.cuda.is_available()
    device = "cuda:5" 

    train_criterion = Criterion(sax=-3.0,saq=-3.0, learn_beta=True)
    val_criterion =Criterion()

    model =EEGNet()
    #EEGNet()
    # ShallowConvNet()
    #DeepConvNet()
    #EEGNet()
    
    param_list = [{'params': model.parameters()}]
    if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
        print('learn_beta')
        param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
    if None is not None and hasattr(train_criterion, 'srx') and hasattr(train_criterion, 'srq'):
        print('learn_gamma')
        param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
    optimizer = torch.optim.Adam(param_list, lr=5e-5, weight_decay=0.0005)



    task="kitti"
    name="kitti"


    if task=="desk01-T500":
        sample_size=500
        n_step = 60
        n_input = 500
        channels = 1
        grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
        rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

        EEG_path="/home/hyx/实验五/Subject05/NP/sub05_seq_500.mat"
        #"/home/hyx/subject5/zqh原始数据/实验1/order1.mat"
        
    if task=="desk01-T200":
        
        sample_size=500
        n_step = 60
        n_input = 200
        channels = 1
        grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
        rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

        EEG_path="/home/hyx/实验五/Subject05/NP/sub05_seq_200.mat"
        #"/home/hyx/subject5/zqh原始数据/实验1/order1.mat"
    if task=="desk01-T50":
        sample_size=500
        n_step = 60
        n_input = 50
        channels = 1
        grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
        rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

        EEG_path="/home/hyx/实验五/Subject01/NP/sub01_seq_50.mat"
        #"/home/hyx/subject5/zqh原始数据/实验1/order1.mat"   
             
    if task=="desk01-T100":
            sample_size=500
            n_step = 60
            n_input = 100
            channels = 1
            grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
            rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

            EEG_path="/home/hyx/实验五/Subject01/NP/sub01_seq_100.mat"
            #"/home/hyx/subject5/zqh原始数据/实验1/order1.mat"   
    if task=="desk01-T300":
            sample_size=500
            n_step = 60
            n_input = 300
            channels = 1
            grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
            rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

            EEG_path="/home/hyx/实验五/Subject05/NP/sub05_seq_300.mat"
            #"/home/hyx/subject5/zqh原始数据/实验1/order1.mat"     
    if task=="desk01-T100-random":
        sample_size=500
        n_step = 60
        n_input = 100
        channels = 1
        grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk/groundtruth.txt'
        rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk/rgb.txt"

        EEG_path="/home/hyx/实验五/Subject01/NP/sub01_random3_100.mat"


    if task=="desk02-T100":
        sample_size=300
        n_step = 60
        n_input = 100
        channels = 1


        EEG_path="/home/hyx/subject3/实验2/原始数据/indoor2.mat"
        #"/home/hyx/subject5/实验2/indoor2.mat"
        # "/home/hyx/subject2/实验2/indoor2.mat" 
        # "/home/hyx/subject3/实验2/ica/indoor2.mat"
        # "/home/hyx/subject4/实验2/ica/indoor2.mat"
        #"/home/hyx/subject5/实验2/indoor2.mat"
        
        grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg1_desk2/groundtruth.txt'
        rgb_txt="/mnt/HYX/rgbd_dataset_freiburg1_desk2/rgb.txt"

    if task=="walking-xyz":
            sample_size=859
            n_step = 60
            n_input = 100
            channels = 1
            EEG_path="/home/hyx/实验4/Dynamic/新实验4/原始数据/qjy_seq01.mat"
            #"/home/hyx/subject2/实验1/order1.mat"
            
            grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt'
            rgb_txt="/mnt/HYX/rgbd_dataset_freiburg3_walking_xyz/rgb.txt"

            #"/home/hyx/subject5/zqh原始数据/实验1/order1.mat"  
    if task=="walking-rpy":
            sample_size=910
            n_step = 60
            n_input = 100
            channels = 1
            EEG_path="/home/hyx/实验4/Dynamic/新实验4/原始数据/qjy_seq02.mat"
            #"/home/hyx/subject2/实验1/order1.mat"
            
            grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/groundtruth.txt'
            rgb_txt="/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/rgb.txt"

    if task=="walking-sphere":
            sample_size=1067
            n_step = 60
            n_input = 100
            channels = 1
            EEG_path="/home/hyx/实验4/Dynamic/新实验4/原始数据/qjy_seq03.mat"
            grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg3_walking_halfsphere/groundtruth.txt'
            rgb_txt="/mnt/HYX/rgbd_dataset_freiburg3_walking_halfsphere/rgb.txt"
    if task=="kitti":

        sample_size=2760
        n_step = 60
        n_input = 100
        channels = 1

        
        # EEG_path="/home/hyx/subject2/实验2/outdoor1.mat"
        # EEG_path1="/home/hyx/subject2/实验2/outdoor2.mat"
        # EEG_path2="/home/hyx/subject2/实验2/outdoor3.mat"
        # EEG_path="/home/hyx/subject4/实验2/原始数据/outdoor1.mat"
        # EEG_path1="/home/hyx/subject4/实验2/原始数据/outdoor2.mat"
        # EEG_path2="/home/hyx/subject4/实验2/原始数据/outdoor3.mat"
        EEG_path="/home/hyx/subject3/实验2/原始数据/outdoor1.mat"
        EEG_path1="/home/hyx/subject3/实验2/原始数据/outdoor2.mat"
        EEG_path2="/home/hyx/subject3/实验2/原始数据/outdoor3.mat"
        # EEG_path="/home/hyx/subject5/zqh原始数据/实验2/outdoor1.mat"
        # EEG_path1="/home/hyx/subject5/zqh原始数据/实验2/outdoor2.mat"
        # EEG_path2="/home/hyx/subject5/zqh原始数据/实验2/outdoor3.mat"
            # print(sio.loadmat(EEG_path))  

        grouthtruth_txt1='/home/hyx/adalashiwork/Kiiti/data_odometry_poses/sequences/05/tum_05_gt.txt'
        eeg_data0 = EEG_reshape(EEG_path)
        eeg_data1 = EEG_reshape(EEG_path1)
        eeg_data2 = EEG_reshape(EEG_path2)
        eeg_data=np.concatenate((eeg_data0,eeg_data1,eeg_data2))
        eeg_poses = read_pose(grouthtruth_txt1)
    if task=="OR-day":
        EEG_test_path = "/home/hyx/实验4/Day/实验4_new/原始数据/day.mat"
        eeg_data = EEG_reshape(EEG_test_path)


        
        
        
        
    if task !="kitti":
        eeg_data = EEG_reshape(EEG_path)
        eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)

    print(sio.loadmat(EEG_path))  
    BATCH_SIZE = 32
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt('/home/hyx/test/BCML/data/Work/SPLIT/train_indices.csv',).astype('int')
    test_indices = np.loadtxt('/home/hyx/test/BCML/data/Work/SPLIT/test_indices.csv').astype('int')
    
    if "random1" in EEG_path:
        folder_path = '/mnt/HYX/rgbd_dataset_freiburg1_desk/random_1'
        pattern = re.compile(r'\((\d+)\)')
        file_names = os.listdir(folder_path)
        sorted_files = sorted(file_names, key=lambda name: int(re.findall(r'\d+', name)[0]) if re.findall(r'\d+', name) else float('inf'))
        indices = []
        for name1 in sorted_files:
            match = pattern.search(name1)
            if match:
                indices.append(int(match.group(1)))
        train_indices = indices[:334]
        test_indices = indices[334:] 
    
    if "random2" in EEG_path:
        folder_path = '/mnt/HYX/rgbd_dataset_freiburg1_desk/random_2'
        pattern = re.compile(r'\((\d+)\)')
        file_names = os.listdir(folder_path)
        sorted_files = sorted(file_names, key=lambda name: int(re.findall(r'\d+', name)[0]) if re.findall(r'\d+', name) else float('inf'))

        indices = []
        for name1 in sorted_files:
            match = pattern.search(name1)
            if match:
                indices.append(int(match.group(1)))
        train_indices = indices[:334]
        test_indices = indices[334:] 
    if "random3" in EEG_path:
        folder_path = '/mnt/HYX/rgbd_dataset_freiburg1_desk/random_3'
        pattern = re.compile(r'\((\d+)\)')
        file_names = os.listdir(folder_path)
        sorted_files = sorted(file_names, key=lambda name: int(re.findall(r'\d+', name)[0]) if re.findall(r'\d+', name) else float('inf'))

        indices = []
        for name1 in sorted_files:
            match = pattern.search(name1)
            if match:
                indices.append(int(match.group(1)))
        train_indices = indices[:334]
        test_indices = indices[334:] 
    
    eeg_y_test=[]
    eeg_y_train=[]
    # eeg_x_test = eeg_data[334:]
    # for i in test_indices:
    #     i=int(i)
    #     eeg_y_test.append(eeg_poses[i-1])

    # eeg_x_train = eeg_data[:334]

    # for i in train_indices:
    #     i=int(i)
    #     eeg_y_train.append(eeg_poses[i-1])

    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]
    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])


    print(train_indices)
    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_train = np.array(eeg_x_train)
    eeg_x_test = np.array(eeg_x_test)
    eeg_y_train = np.array(eeg_y_train)
    eeg_y_test = np.array(eeg_y_test)
    print(eeg_x_train.shape)
    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1),torch.tensor(eeg_y_train))
    train_dataloader =Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1),torch.tensor(eeg_y_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)





    model.to(device)
    train_criterion.to(device)
    val_criterion.to(device)
    torch.cuda.is_available()
    val_loader=test_dataloader
    train_loss=[]


    save_dir = "/home/hyx/test/BCML/SAVE_EEG/NC/T100"
    for e in range(3000):
        model.train()
        for batch in train_dataloader:
            
            model.zero_grad()
            eeg, label = batch
            batch_size = eeg.size(0)
            #log q  [y_tilde_eeg,y_tilde_image,label]
            y=np.zeros((len(label), 6))
            for i in range(len(y)):
                p = label[i, :3]  
                q = label[i, 3:7]
                q *= np.sign(q[0])  # constrain to hemisphere
                back=qlog(q)
                y[i] = np.hstack((p, back))

            y=torch.tensor(y)
            eeg = to_gpu(eeg)
            Label = to_gpu(y)
            y_tilde_eeg = model(eeg)
            y_tilde_eeg=to_gpu(y_tilde_eeg)


            loss_tmp = train_criterion(y_tilde_eeg, Label)
            loss_tmp.backward()        
            optimizer.step()
            optimizer.zero_grad()        
            train_loss.append(loss_tmp.item())
        ##print("train_loss: {:.3f}".format(loss_tmp))
        print("Epoch {0}: loss={1:.4f}, ".format(e, np.mean(train_loss)))
           

        L = int(sample_size//3)



        predeg_poses = np.zeros((L, 7))  # store all predicted poses
        predim_poses = np.zeros((L, 7))
        targ_poses = np.zeros((L, 7))  # store all target poses 
        pose_stats_file = osp.join('/home/hyx/test/BCML/data/Work/pose_stats.txt')
        pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
        t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        q_criterion = quaternion_angular_error
        y_true, y_pred_eeg, y_pred_image = [], [], []
        eval_loss_eeg, eval_loss_image = [], []

        val_loss = AverageMeter()
        model.eval()             #设置为评估模式
        with torch.no_grad():
            for batch in test_dataloader:
                model.zero_grad()
                eeg, label = batch
                y=np.zeros((len(label), 6))
                for i in range(len(y)):
                    p = label[i, :3]  
                    q = label[i, 3:7]
                    q *= np.sign(q[0])  # constrain to hemisphere
                    back=qlog(q)
                    y[i] = np.hstack((p, back))  
                y=torch.tensor(y)
                eeg = to_gpu(eeg)
                y = to_gpu(y)
                y_tilde_eeg = model(eeg)
                y_tilde_eeg=to_gpu(y_tilde_eeg)
                eval_loss=val_criterion(y_tilde_eeg, y)
                y_pred_eeg.append(y_tilde_eeg.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy()) 
            eval_loss_eeg.append(eval_loss.item())

                #val_output=y_tilde_eeg
                #val_target=y
                #s = val_output.size()

            if len(y_true)==L:
                y_true = np.concatenate(y_true, axis=0).squeeze()
                y_pred_eeg = np.concatenate(y_pred_eeg, axis=0).squeeze()
                p,q=[],[]
                for i in range(len(y_true)):
                    s = torch.Size([1, 6])
                    eeg_output = y_pred_eeg[i].reshape((-1, s[-1]))
                    target = y_true[i].reshape((-1, s[-1]))
                    q = [qexp(p[3:]) for p in eeg_output]
                    eeg_output = np.hstack((eeg_output[:, :3], np.asarray(q)))
                    q = [qexp(p[3:]) for p in target]
                    target = np.hstack((target[:, :3], np.asarray(q)))
                    eeg_output[:, :3] = (eeg_output[:, :3] * pose_s) + pose_m
                    target[:, :3] = (target[:, :3] * pose_s) + pose_m
                # take the middle prediction
                    predeg_poses[i, :] = eeg_output[len(eeg_output) // 2]
                    targ_poses[i, :] = target[len(target) // 2]
        eeg_t_loss = np.asarray([t_criterion(p, t) for p, t in zip(predeg_poses[:, :3], targ_poses[:, :3])])
        eeg_q_loss = np.asarray([q_criterion(p, t) for p, t in zip(predeg_poses[:, 3:], targ_poses[:, 3:])])
        mean_t_loss = eeg_t_loss.mean()
        save_dir = '/home/hyx/test/BCML/memory_EEG'
        if e==2999:

            os.makedirs(save_dir, exist_ok=True)

            pred_path = os.path.join(save_dir, f'{name}_predicted_poses.txt')
            targ_path = os.path.join(save_dir, f'{name}_target_poses.txt')

            np.savetxt(pred_path, predeg_poses, fmt='%.6f')
            np.savetxt(targ_path, targ_poses, fmt='%.6f')

        eval_result_filename = osp.join(save_dir, f'{name}.txt')
        with open(eval_result_filename, 'a') as f:
            f.write(f"Epoch {e}: \n")
            f.write(f'Translation error: median={np.median(eeg_t_loss):.2f} m, mean={np.mean(eeg_t_loss):.2f} m, std={np.std(eeg_t_loss):.2f} m, var={np.var(eeg_t_loss):.4f} m^2\n')
            f.write(f'Rotation error:    median={np.median(eeg_q_loss):.2f} deg, mean={np.mean(eeg_q_loss):.2f} deg, std={np.std(eeg_q_loss):.2f} deg, var={np.var(eeg_q_loss):.4f} deg^2\n')



        real_pose = (predeg_poses[:, :3] - pose_m) / pose_s
        gt_pose = (targ_poses[:, :3] - pose_m) / pose_s

        # ========== XY 图 ==========
        # fig_xy, ax_xy = plt.subplots(figsize=(8, 6))
        # ax_xy.plot(gt_pose[:, 0], gt_pose[:, 1], 'ko', label='Ground Truth Pose')
        # ax_xy.plot(real_pose[:, 0], real_pose[:, 1], 'rx', label='Predicted Pose')

        # for i in range(len(gt_pose)):
        #     ax_xy.plot([gt_pose[i, 0], real_pose[i, 0]], [gt_pose[i, 1], real_pose[i, 1]], 'gray', linestyle='--', alpha=0.5)

        # ax_xy.set_xlabel('x [m]')
        # ax_xy.set_ylabel('y [m]')
        # ax_xy.legend()
        # ax_xy.grid(True)
        # fig_xy.tight_layout()
        # xy_filename = osp.join(save_dir, f'{name}.EEG_xy.png')
        # fig_xy.savefig(xy_filename, dpi=300)
        # print(xy_filename)

        # # ========== XZ 图 ==========
        # fig_xz, ax_xz = plt.subplots(figsize=(8, 6))
        # ax_xz.plot(gt_pose[:, 0], gt_pose[:, 2], 'ko', label='Ground Truth Pose')
        # ax_xz.plot(real_pose[:, 0], real_pose[:, 2], 'rx', label='Predicted Pose')

        # for i in range(len(gt_pose)):
        #     ax_xz.plot([gt_pose[i, 0], real_pose[i, 0]], [gt_pose[i, 2], real_pose[i, 2]], 'gray', linestyle='--', alpha=0.5)

        # ax_xz.set_xlabel('x [m]')
        # ax_xz.set_ylabel('z [m]')
        # ax_xz.legend()
        # ax_xz.grid(True)
        # fig_xz.tight_layout()
        # xz_filename = osp.join(save_dir, f'{name}.EEG_xz.png')
        # # fig_xz.savefig(xz_filename, dpi=300)
        # print(xz_filename)
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(111, projection='3d')
        # ax2.plot(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], color='black', marker='o', alpha=1,linestyle='', label='Ground Truth Pose')
        # ax2.plot(real_pose[:, 0], real_pose[:, 1], real_pose[:, 2], color='red', marker='s', alpha=0.5,linestyle='', label='Predicted Pose')
        # for i in range(len(gt_pose)):
        #     ax2.plot([gt_pose[i, 0], real_pose[i, 0]],
        #             [gt_pose[i, 1], real_pose[i, 1]],
        #             [gt_pose[i, 2], real_pose[i, 2]],
        #             color='gray', linestyle='--', alpha=0.5)
        # ax2.set_xlabel('x [m]')
        # ax2.set_ylabel('y [m]')
        # ax2.set_zlabel('z [m]')
        # ax2.legend()
        # image_filename2 = osp.join(save_dir, f'{name}-EEG_2-3d.png')
        # fig2.savefig(image_filename2)
        # print(f"3D 图像保存至: {image_filename2}")
            
        # # ========== 3D 图 ==========
        # fig3d = plt.figure(figsize=(10, 8))
        # ax3d = fig3d.add_subplot(111, projection='3d')
        # ax3d.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], color='black', marker='o', s=30, alpha=0.8, label='Ground Truth Pose')
        # ax3d.scatter(real_pose[:, 0], real_pose[:, 1], real_pose[:, 2], color='red', marker='^', s=30, alpha=0.6, label='Predicted Pose')

        # for i in range(len(gt_pose)):
        #     ax3d.plot([gt_pose[i, 0], real_pose[i, 0]],
        #             [gt_pose[i, 1], real_pose[i, 1]],
        #             [gt_pose[i, 2], real_pose[i, 2]],
        #             color='gray', linestyle='--', alpha=0.4, linewidth=1.0)

        # # 设置坐标范围（含边距）
        # x_margin = (gt_pose[:, 0].ptp()) * 0.1
        # y_margin = (gt_pose[:, 1].ptp()) * 0.1
        # z_margin = (gt_pose[:, 2].ptp()) * 0.1
        # ax3d.set_xlim(gt_pose[:, 0].max() + x_margin, gt_pose[:, 0].min() - x_margin)
        # ax3d.set_ylim(gt_pose[:, 1].max() + y_margin, gt_pose[:, 1].min() - y_margin)
        # ax3d.set_zlim(gt_pose[:, 2].min() - z_margin, gt_pose[:, 2].max() + z_margin)

        # ax3d.set_xlabel('x [m]')
        # ax3d.set_ylabel('y [m]')
        # ax3d.set_zlabel('z [m]')
        # ax3d.view_init(elev=25, azim=135)
        # ax3d.legend()
        # ax3d.grid(True)
        # fig3d.tight_layout()
        # pose3d_filename = osp.join(save_dir, f'{name}.EEG_3d.png')
        # fig3d.savefig(pose3d_filename, dpi=300)
        # print(pose3d_filename)
        # plt.close(fig_xy)
        # plt.close(fig_xz)
        # plt.close(fig2)
        # plt.close(fig3d)




















