# Imports
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD, MMD_loss, get_shuffled,Criterion, loss_fn,qlog,apply_qlog_to_tensor,qexp,quaternion_angular_error
import model

from sklearn.metrics import mean_squared_error 
import os.path as osp
from torchvision import transforms, models


import os
# BMCL framework
class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):
        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        self.train_criterion = Criterion(sax=self.train_config.sax,saq=self.train_config.saq, learn_beta=True)
        #self.train_criterion1 = Criterion(sax=self.train_config.sax1,saq=self.train_config.saq1, learn_beta=True)
    
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(model, self.train_config.model)(self.train_config)
        # Final list
        for name, param in self.model.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)
        # if torch.cuda.is_available() and cuda:
        #     self.model.cuda()

        self.param_list = [{'params': self.model.parameters()}]
        if hasattr(self.train_config, 'sax') and hasattr(self.train_config, 'saq'):
            print('learn_beta')
            self.param_list.append({'params': [self.train_criterion.sax, self.train_criterion.saq]})
            #self.param_list.append({'params': [self.train_criterion1.sax, self.train_criterion1.saq]})


        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                self.param_list,
                #filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

    @time_desc_decorator('Training Start!')
    def train(self):
        self.name="test"
        #"subject02-desk01-T500-default"
        self.criterion1=criterion=self.train_criterion
       # self.criterion2=criterion1=self.train_criterion1
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_sim = nn.TripletMarginLoss(margin=1.0, p=2.0) #nn.MSELoss() #MMD_loss() #CMD()
        self.L=int(500//3)
        #self.L=int(2760//3)
        #self.L=int(859//3)
        #self.L=int(910//3) 
        #self.L=int(1067 //3)     
        # self.L=int(4200*0.3)
        # self.L=int(3700*0.3)
        train_losses = []

        eeg_best_tloss = float(50)  
        eeg_best_qloss = float('inf')  
        image_best_tloss = float(50)  
        image_best_qloss = float('inf')  
        e1,e2,e3,e4=0,0,0,0

        for e in range(self.train_config.n_epoch):
            self.model=to_gpu(self.model)
            self.model.train()
            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss = []
            
            for batch in self.train_data_loader:
                self.model.zero_grad()
                eeg, label, image = batch
                #y=label
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
                image = to_gpu(image)
                label = to_gpu(y)
                y_tilde_eeg, y_tilde_image = self.model(eeg, image)
                #label=to_gpu(label1)
                y_tilde_eeg=to_gpu(y_tilde_eeg)
                y_tilde_image=to_gpu(y_tilde_image)
                cls_loss =criterion(y_tilde_eeg, label) +criterion(y_tilde_image, label) 

                diff_loss = self.get_diff_loss()
                sim_loss = self.get_sim_loss(label)

                if self.train_config.use_sim:
                    similarity_loss = sim_loss
                loss = cls_loss * self.train_config.cla_weight + self.train_config.diff_weight * diff_loss + self.train_config.sim_weight * similarity_loss 
                loss.backward()
                #torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss.append(loss.item())
                train_loss_sim.append(similarity_loss.item())
            train_losses.append(train_loss)
            print("Epoch {0}: loss={1:.4f}".format(e, np.mean(train_loss)))
            model_params = {
                'project_e': self.model.project_e.state_dict(),
                'private_e':self.model.private_e.state_dict(),
                'common': self.model.common.state_dict(),
                'private_e_2':self.model.private_e_2.state_dict(),
                'common_2': self.model.common_2.state_dict(),
                'criterion_params': {
                'sax': self.train_criterion.sax.data.clone(),
                'saq': self.train_criterion.saq.data.clone(),
                }
                # 'private_e_3':self.model.private_e_3.state_dict(),
                # 'common_3': self.model.common_3.state_dict(),
            }
            save_path="/home/hyx/test/BCML/models/"+f"{self.name}model_params.pth"
            torch.save(model_params, save_path)
            print(f"模型参数已保存至 {save_path}")


            if e % 10 == 0:  # 每10个epoch
                valid_loss_eeg, valid_loss_image, _, _, im_tloss, im_qloss = self.eval(mode="dev")
                _, eeg_tloss, eeg_qloss, eeg_tvariance, eeg_tstd, eeg_qvariance, eeg_qstd,_,_ = self.eval_eeg(mode="dev", to_print=False, plot=False)

                print("当前最佳 EEG tloss: {:.4f} 在 epoch:{:.1f}, qloss: {:.4f} 在 epoch:{:.1f}".format(eeg_best_tloss, e1, eeg_best_qloss, e2))
                print("当前最佳 IM tloss: {:.4f} 在 epoch:{:.1f}, qloss: {:.4f} 在 epoch:{:.1f}".format(image_best_tloss, e3, image_best_qloss, e4))

                # 打印标准差和方差
                print("当前 EEG tloss 标准差: {:.4f}, 方差: {:.4f}".format(eeg_tstd, eeg_tvariance))
                print("当前 EEG qloss 标准差: {:.4f}, 方差: {:.4f}".format(eeg_qstd, eeg_qvariance))

                #test
                with open(f'/home/hyx/test/BCML/SAVE/{self.name}.txt', 'a') as log_file:
                    log_file.write("Epoch {}: 当前最佳 EEG tloss: {:.4f}, qloss: {:.4f}, IM tloss: {:.4f}, qloss: {:.4f}\n".format(
                        e, eeg_best_tloss, eeg_best_qloss, image_best_tloss, image_best_qloss))
                    log_file.write("Epoch {}: EEG tloss: {:.4f}, EEG qloss: {:.4f}, IM tloss: {:.4f}, IM qloss: {:.4f}\n".format(
                        e, eeg_tloss, eeg_qloss, im_tloss, im_qloss))
                    log_file.write("Epoch {}: EEG tloss 标准差: {:.4f}, 方差: {:.4f}, EEG qloss 标准差: {:.4f}, 方差: {:.4f}\n".format(
                        e, eeg_tstd, eeg_tvariance, eeg_qstd, eeg_qvariance))
            else:
                valid_loss_eeg, valid_loss_image, _, _, im_tloss, im_qloss = self.eval(mode="dev")
                _, eeg_tloss, eeg_qloss, eeg_tvariance, eeg_tstd, eeg_qvariance, eeg_qstd,_,_ = self.eval_eeg(mode="dev", to_print=False, plot=False)
                with open(f'/home/hyx/test/BCML/SAVE/{self.name}.txt', 'a') as log_file:
                    log_file.write("Epoch {}: EEG tloss: {:.4f}, EEG qloss: {:.4f}, IM tloss: {:.4f}, IM qloss: {:.4f}\n".format(
                        e, eeg_tloss, eeg_qloss, im_tloss, im_qloss))
                    log_file.write("Epoch {}: EEG tloss 标准差: {:.4f}, 方差: {:.4f}, EEG qloss 标准差: {:.4f}, 方差: {:.4f}\n".format(
                        e, eeg_tstd, eeg_tvariance, eeg_qstd, eeg_qvariance))
            if eeg_tloss < eeg_best_tloss:
                eeg_best_tloss = eeg_tloss 
                e1 = e 
                #valid_loss_eeg, valid_loss_image, eeg_tloss, eeg_qloss, im_tloss, im_qloss = self.eval(mode="dev",to_print="EEG",plot="2D")
                _, eeg_tloss, eeg_qloss, eeg_tvariance, eeg_tstd, eeg_qvariance, eeg_qstd,predeg_poses,targ_poses = self.eval_eeg(mode="dev", to_print=True, plot=False)
                

                save_dir = '/home/hyx/test/BCML/memory'
                os.makedirs(save_dir, exist_ok=True)

                # 构建完整的文件路径
                pred_path = os.path.join(save_dir, f'{self.name}_predicted_poses.txt')
                targ_path = os.path.join(save_dir, f'{self.name}_target_poses.txt')

                # 保存
                np.savetxt(pred_path, predeg_poses, fmt='%.6f')
                np.savetxt(targ_path, targ_poses, fmt='%.6f')
            if eeg_qloss < eeg_best_qloss:  
                eeg_best_qloss = eeg_qloss
                e2 = e  

            if im_tloss < image_best_tloss:  
                image_best_tloss = im_tloss 
                e3 = e
                #valid_loss_eeg, valid_loss_image, eeg_tloss, eeg_qloss, im_tloss, im_qloss = self.eval(mode="dev",to_print="IM",plot="2D")

            if im_qloss < image_best_qloss:  
                image_best_qloss = im_qloss
                e4 = e  



    def eval(self,mode=None, to_print=None,plot=None):
        self.model=self.model.to("cpu")
        eval_model=self.model
        eval_model.image_model.droprate = 0
        eval_model==to_gpu(eval_model)
        assert(mode is not None)
        eval_model.eval()
        y_true, y_pred_eeg, y_pred_image = [], [], []
        eval_loss_eeg, eval_loss_image = [], []
        L = self.L
        predeg_poses = np.zeros((L, 7))  # store all predicted poses
        predim_poses = np.zeros((L, 7))
        targ_poses = np.zeros((L, 7))  # store all target poses 
        pose_stats_file = osp.join('/home/hyx/test/BCML/data/Work/pose_stats.txt')
        pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
        if self.name=="day":
            pose_stats_file = osp.join('/home/hyx//data/RobotCar/full/pose_stats.txt')
            pose_m, pose_s = np.loadtxt(pose_stats_file) 
        if self.name=="night":
            pose_stats_file = osp.join('/home/hyx//data/RobotCar/full/pose_stats.txt')
            pose_m, pose_s = np.loadtxt(pose_stats_file) 
        t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        q_criterion = quaternion_angular_error
        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "train":
            dataloader = self.train_data_loader
        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                eeg, label, image = batch
                #y=label
                y=np.zeros((len(label), 6))
                for i in range(len(y)):
                    p = label[i, :3]  
                    q = label[i, 3:7]
                    q *= np.sign(q[0])  # constrain to hemisphere
                    back=qlog(q)
                    y[i] = np.hstack((p, back))  
                y=torch.tensor(y)
                eeg = to_gpu(eeg)
                image = to_gpu(image)
                y = to_gpu(y)
                y_tilde_eeg, y_tilde_image = eval_model(eeg, image)
                cls_loss_eeg = self.criterion1(y_tilde_eeg, y)
                cls_loss_image = self.criterion1(y_tilde_image, y)
                loss_eeg = cls_loss_eeg
                loss_image = cls_loss_image
                eval_loss_eeg.append(loss_eeg.item())
                eval_loss_image.append(loss_image.item())
                y_pred_eeg.append(y_tilde_eeg.detach().cpu().numpy())
                y_pred_image.append(y_tilde_image.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())        
                #eval pose  
                if len(y_true)==L:
                    y_true = np.concatenate(y_true, axis=0).squeeze()
                    y_pred_eeg = np.concatenate(y_pred_eeg, axis=0).squeeze()
                    y_pred_image = np.concatenate(y_pred_image, axis=0).squeeze()
                    p,q=[],[]
                    for i in range(len(y_true)):
                        s = torch.Size([1, 6])
                        eeg_output = y_pred_eeg[i].reshape((-1, s[-1]))
                        image_output=y_pred_image[i].reshape((-1, s[-1]))
                        target = y_true[i].reshape((-1, s[-1]))
                        q = [qexp(p[3:]) for p in eeg_output]
                        eeg_output = np.hstack((eeg_output[:, :3], np.asarray(q)))
                        q = [qexp(p[3:]) for p in image_output]
                        image_output = np.hstack((image_output[:, :3], np.asarray(q)))
                        q = [qexp(p[3:]) for p in target]
                        target = np.hstack((target[:, :3], np.asarray(q)))
                        eeg_output[:, :3] = (eeg_output[:, :3] * pose_s) + pose_m
                        image_output[:, :3] = (image_output[:, :3] * pose_s) + pose_m
                        target[:, :3] = (target[:, :3] * pose_s) + pose_m
                    # take the middle prediction
                        predeg_poses[i, :] = eeg_output[len(eeg_output) // 2]
                        predim_poses[i, :] = image_output[len(image_output) // 2]
                        targ_poses[i, :] = target[len(target) // 2]
        eeg_t_loss = np.asarray([t_criterion(p, t) for p, t in zip(predeg_poses[:, :3], targ_poses[:, :3])])
        eeg_q_loss = np.asarray([q_criterion(p, t) for p, t in zip(predeg_poses[:, 3:], targ_poses[:, 3:])])
        im_t_loss = np.asarray([t_criterion(p, t) for p, t in zip(predim_poses[:, :3], targ_poses[:, :3])])
        im_q_loss = np.asarray([q_criterion(p, t) for p, t in zip(predim_poses[:, 3:], targ_poses[:, 3:])])
        eval_loss_eeg = np.mean(eval_loss_eeg)
        eval_loss_image = np.mean(eval_loss_image)
        if plot=="2D":
            if to_print == "EEG":  
                    # 标准化姿态  
                realeg_pose = (predeg_poses[:, :3] - pose_m) / pose_s  
                realim_pose = (predim_poses[:, :3] - pose_m) / pose_s  
                gt_pose = (targ_poses[:, :3] - pose_m) / pose_s  
    
                    # 绘制第一个图形：realeg_pose  
                fig1, ax1 = plt.subplots()  
                ax1.plot(gt_pose[:, 0], gt_pose[:, 1], color='black', marker='o', linestyle='', label='Ground Truth Pose')  
                ax1.plot(realeg_pose[:, 0], realeg_pose[:, 1], color='red', marker='s', linestyle='', label='Predicted Pose (predeg_poses)')  
                for i in range(len(gt_pose)): 
                    ax1.plot([gt_pose[i, 0], realeg_pose[i, 0]], [gt_pose[i, 1], realeg_pose[i, 1]], color='gray', linestyle='--', alpha=0.5)  
            
                ax1.set_xlabel('x [m]')  
                ax1.set_ylabel('y [m]')  
                ax1.legend()  
                ax1.grid(True)  
                image_filename1 = osp.join(osp.expanduser("/home/hyx/test/BCML/SAVE"), f'{self.name}-EEG.png')  
                fig1.savefig(image_filename1)  
                print(image_filename1)  
            if to_print == "IM":
                realeg_pose = (predeg_poses[:, :3] - pose_m) / pose_s  
                realim_pose = (predim_poses[:, :3] - pose_m) / pose_s  
                gt_pose = (targ_poses[:, :3] - pose_m) / pose_s  
    
                # 绘制第二个图形：realim_pose（类似地）  
                fig2, ax2 = plt.subplots()  
                ax2.plot(gt_pose[:, 0], gt_pose[:, 1], color='black', marker='o', linestyle='', label='Ground Truth Pose')  
                ax2.plot(realim_pose[:, 0], realim_pose[:, 1], color='blue', marker='s', linestyle='', label='Predicted Pose (predim_poses)')  
                # ax2.plot(gt_pose[40:, 0], gt_pose[40:, 1], color='black', marker='o', linestyle='', label='Ground Truth Pose')  
                # ax2.plot(realim_pose[40:, 0], realim_pose[40:, 1], color='blue', marker='s', linestyle='', label='Predicted Pose (predim_poses)')  
            
                # 假设gt_pose和realim_pose长度相同且顺序对应，添加连线 
                # for i in range(len(gt_pose)):  
                for i in range(len(gt_pose)): 
                    ax2.plot([gt_pose[i, 0], realim_pose[i, 0]], [gt_pose[i, 1], realim_pose[i, 1]], color='gray', linestyle='--', alpha=0.5)  
            
                ax2.set_xlabel('x [m]')  
                ax2.set_ylabel('y [m]')  
                ax2.legend()  
                ax2.grid(True)  
                image_filename2= osp.join(osp.expanduser("/home/hyx/test/BCML/SAVE"), f'{self.name}-IM.png')  
                fig2.savefig(image_filename2)  
                print(image_filename2)
                plt.close()
        if plot=="3D":
            if to_print == "EEG":
                # 标准化姿态
                realeg_pose = (predeg_poses[:, :3] - pose_m) / pose_s
                realim_pose = (predim_poses[:, :3] - pose_m) / pose_s
                gt_pose = (targ_poses[:, :3] - pose_m) / pose_s

                # 绘制第一个图形：realeg_pose (3D图)
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111, projection='3d')  # 创建3D坐标轴

                ax1.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], color='black', marker='o', label='Ground Truth Pose', s=7, alpha=0.8)
                ax1.scatter(realeg_pose[:, 0], realeg_pose[:, 1], realeg_pose[:, 2], color='red', marker='s', label='Predicted Pose (predeg_poses)', s=5,alpha=0.5)

                # 假设gt_pose和realeg_pose长度相同且顺序对应，添加连线
                for i in range(len(gt_pose)):
                    ax1.plot([gt_pose[i, 0], realeg_pose[i, 0]], [gt_pose[i, 1], realeg_pose[i, 1]], [gt_pose[i, 2], realeg_pose[i, 2]], color='gray', linestyle='--', alpha=0.5)

                ax1.set_xlabel('x [m]')
                ax1.set_ylabel('y [m]')
                ax1.set_zlabel('z [m]')
                ax1.legend()
                ax1.grid(True)

                image_filename1 = osp.join(osp.expanduser("/home/hyx/test/BCML/checkpoint"), 'EEG_desk01_04.png')
                fig1.savefig(image_filename1)
                print(image_filename1)

            if to_print == "IM":
                # 标准化姿态
                realeg_pose = (predeg_poses[:, :3] - pose_m) / pose_s
                realim_pose = (predim_poses[:, :3] - pose_m) / pose_s
                gt_pose = (targ_poses[:, :3] - pose_m) / pose_s

                # 绘制第二个图形：realim_pose (3D图)
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111, projection='3d')  # 创建3D坐标轴

                ax2.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], color='black', marker='o', label='Ground Truth Pose', s=7, alpha=0.8)
                ax2.scatter(realim_pose[:, 0], realim_pose[:, 1], realim_pose[:, 2], color='blue', marker='s', label='Predicted Pose (predim_poses)', s=7,alpha=0.45)

                # 假设gt_pose和realim_pose长度相同且顺序对应，添加连线
                for i in range(len(gt_pose)):
                    ax2.plot([gt_pose[i, 0], realim_pose[i, 0]], [gt_pose[i, 1], realim_pose[i, 1]], [gt_pose[i, 2], realim_pose[i, 2]], color='gray', linestyle='--', alpha=0.5)

                ax2.set_xlabel('x [m]')
                ax2.set_ylabel('y [m]')
                ax2.set_zlabel('z [m]')
                ax2.legend()
                ax2.grid(True)

                image_filename2 = osp.join(osp.expanduser("/home/hyx/test/BCML/checkpoint"), 'IM_desk01_04.png')
                fig2.savefig(image_filename2)
                print(image_filename2)
                plt.close()
        return eval_loss_eeg, eval_loss_image, np.mean(eeg_t_loss),np.mean(eeg_q_loss),np.mean(im_t_loss),np.mean(im_q_loss)
    

    def eval_eeg(self, mode=None, to_print=True, plot=True):
        self.model = self.model.to("cpu")
        eval_model = self.model
        model_params_path = "/home/hyx/test/BCML/models/"+f"{self.name}model_params.pth"
        model_params = torch.load(model_params_path)
        eval_model.project_e.load_state_dict(model_params['project_e'])
        eval_model.private_e.load_state_dict(model_params['private_e'])
        # eval_model.attn_fusion.load_state_dict(model_params['attn_fusion'])
        eval_model.common.load_state_dict(model_params['common'])
        eval_model.private_e_2.load_state_dict(model_params['private_e_2'])
        eval_model.common_2.load_state_dict(model_params['common_2'])
        # eval_model.private_e_3.load_state_dict(model_params['private_e_3'])
        # eval_model.common_3.load_state_dict(model_params['common_3'])
        eval_model = to_gpu(eval_model)
        assert mode is not None
        eval_model.eval()
        y_true, y_pred_eeg = [], []
        eval_loss_eeg = []
        L = self.L
        predeg_poses = np.zeros((L, 7)) 
        targ_poses = np.zeros((L, 7))  
        pose_stats_file = osp.join('/home/hyx/test/BCML/data/Work/pose_stats.txt')
        pose_m, pose_s = np.loadtxt(pose_stats_file)  
        if self.name=="day":
            pose_stats_file = osp.join('/home/hyx/adalashiwork/AtLoc-master/data/RobotCar/full/pose_stats.txt')
            pose_m, pose_s = np.loadtxt(pose_stats_file) 
        if self.name=="night":
            pose_stats_file = osp.join('/home/hyx/adalashiwork/AtLoc-master/data/RobotCar/full/pose_stats.txt')
            pose_m, pose_s = np.loadtxt(pose_stats_file)   
        t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        q_criterion = quaternion_angular_error

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "train":
            dataloader = self.train_data_loader

        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                eeg, label, _ = batch  

                y = np.zeros((len(label), 6))
                for i in range(len(y)):
                    p = label[i, :3]  
                    q = label[i, 3:7]  
                    q *= np.sign(q[0])  
                    back = qlog(q)
                    y[i] = np.hstack((p, back))  

                y = torch.tensor(y)
                if self.name=="day":
                    y=label
                if self.name=="night":
                    y=label
                eeg = to_gpu(eeg)
                y = to_gpu(y)


                representation_eeg = eval_model.eeg_model(eeg) 
                # representation_eeg_channel,representation_eeg_time=eval_model.eeg_model(eeg)
                # representation_eeg, attn_weights = eval_model.attn_fusion(representation_eeg_channel,representation_eeg_time)
                representation_e = eval_model.project_e(representation_eeg)
                representation_private_e = eval_model.private_e(representation_e)
                representation_common_e = eval_model.common(representation_e)
                
                rep_e=torch.stack((representation_private_e, representation_common_e), dim=0)
                rep_e=torch.cat((rep_e[0], rep_e[1]), dim=1)
                representation_common_e= eval_model.common_2(rep_e)
                representation_private_e=eval_model.private_e_2(rep_e)
                
                # rep_e=torch.stack((representation_private_e, representation_common_e), dim=0)
                # rep_e=torch.cat((rep_e[0], rep_e[1]), dim=1)
                # representation_common_e= eval_model.common_3(rep_e)
                # representation_private_e=eval_model.private_e_3(rep_e)
                
                h_e = torch.stack((representation_private_e, representation_common_e), dim=0)
                h_e = torch.cat((h_e[0], h_e[1]), dim=1)  # 合并 private 和 common 表示
                y_tilde_eeg = eval_model.fusion_e_xyz(h_e)
                cls_loss_eeg = self.criterion1(y_tilde_eeg, y)
                loss_eeg = cls_loss_eeg
                eval_loss_eeg.append(loss_eeg.item())

                y_pred_eeg.append(y_tilde_eeg.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

                if len(y_true) == L:
                    y_true = np.concatenate(y_true, axis=0).squeeze()
                    y_pred_eeg = np.concatenate(y_pred_eeg, axis=0).squeeze()

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


                        predeg_poses[i, :] = eeg_output[len(eeg_output) // 2]
                        targ_poses[i, :] = target[len(target) // 2]




        eeg_t_loss = np.asarray([t_criterion(p, t) for p, t in zip(predeg_poses[:, :3], targ_poses[:, :3])])
        eeg_q_loss = np.asarray([q_criterion(p, t) for p, t in zip(predeg_poses[:, 3:], targ_poses[:, 3:])])
        
        eval_loss_eeg = np.mean(eval_loss_eeg)

        eeg_t_mean = np.mean(eeg_t_loss)
        eeg_t_variance = np.var(eeg_t_loss)
        eeg_t_std = np.std(eeg_t_loss)
        
        eeg_q_mean = np.mean(eeg_q_loss)
        eeg_q_variance = np.var(eeg_q_loss)
        eeg_q_std = np.std(eeg_q_loss)

        # if to_print:
        #     print(f"EEG 模型评估损失: {eval_loss_eeg:.4f}")
        #     print(f"位置误差（m）：{np.mean(eeg_t_loss):.4f}")
        #     print(f"姿态误差（弧度）：{np.mean(eeg_q_loss):.4f}")

        if plot:
            realeg_pose = (predeg_poses[:, :3] - pose_m) / pose_s  
            gt_pose = (targ_poses[:, :3] - pose_m) / pose_s
              
            # save_path_realeg = "/home/hyx/test/BCML/realeg_pose.npy"
            # save_path_gt = "/home/hyx/test/BCML/gt_pose.npy"
            
            # # 保存数据
            # np.save(save_path_realeg, realeg_pose)
            # np.save(save_path_gt, gt_pose)
            # # ---- 2D 绘图 (x-y 平面) ----
            # fig1, ax1 = plt.subplots()
            # ax1.plot(gt_pose[:, 0], gt_pose[:, 1], color='black', marker='o', linestyle='', label='Ground Truth Pose')
            # ax1.plot(realeg_pose[:, 0], realeg_pose[:, 1], color='red', marker='s', linestyle='', label='Predicted Pose')
            # for i in range(len(gt_pose)):
            #     ax1.plot([gt_pose[i, 0], realeg_pose[i, 0]], [gt_pose[i, 1], realeg_pose[i, 1]], color='gray', linestyle='--', alpha=0.5)
            # ax1.set_xlabel('x [m]')
            # ax1.set_ylabel('y [m]')
            # ax1.legend()
            # ax1.grid(True)
            # image_filename1 = osp.join(osp.expanduser("/home/hyx/test/BCML/SAVE"), f'{self.name}-EEG_only_2d.png')
            # fig1.savefig(image_filename1)
            # print(f"2D 图像保存至: {image_filename1}")
            
            
            
            
            
            
            
            

            # ---- 3D 绘图 (x-y-z 空间) ----
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.plot(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], color='black', marker='o', alpha=1,linestyle='', label='Ground Truth Pose')
            ax2.plot(realeg_pose[:, 0], realeg_pose[:, 1], realeg_pose[:, 2], color='red', marker='s', alpha=0.5,linestyle='', label='Predicted Pose')
            for i in range(len(gt_pose)):
                ax2.plot([gt_pose[i, 0], realeg_pose[i, 0]],
                        [gt_pose[i, 1], realeg_pose[i, 1]],
                        [gt_pose[i, 2], realeg_pose[i, 2]],
                        color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('x [m]')
            ax2.set_ylabel('y [m]')
            ax2.set_zlabel('z [m]')
            ax2.legend()
            image_filename2 = osp.join(osp.expanduser("/home/hyx/test/BCML/SAVE"), f'{self.name}-EEG_3d.png')
            fig2.savefig(image_filename2)
            print(f"3D 图像保存至: {image_filename2}")
            
            
            fig2 = plt.figure(figsize=(10, 8))  
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2],
                        color='black', marker='o', s=30, alpha=0.8, label='Ground Truth Pose')
            ax2.scatter(realeg_pose[:, 0], realeg_pose[:, 1], realeg_pose[:, 2],
                        color='red', marker='^', s=30, alpha=0.6, label='Predicted Pose')
            for i in range(len(gt_pose)):
                ax2.plot([gt_pose[i, 0], realeg_pose[i, 0]],
                        [gt_pose[i, 1], realeg_pose[i, 1]],
                        [gt_pose[i, 2], realeg_pose[i, 2]],
                        color='gray', linestyle='--', alpha=0.4, linewidth=1.0)
            ax2.set_xlabel('x [m]')
            ax2.set_ylabel('y [m]')
            ax2.set_zlabel('z [m]')
            ax2.view_init(elev=25, azim=135)  
            x_margin = (gt_pose[:, 0].max() - gt_pose[:, 0].min()) * 0.1
            y_margin = (gt_pose[:, 1].max() - gt_pose[:, 1].min()) * 0.1
            z_margin = (gt_pose[:, 2].max() - gt_pose[:, 2].min()) * 0.1
            ax2.set_xlim(gt_pose[:, 0].max() + x_margin, gt_pose[:, 0].min() - x_margin)
            ax2.set_ylim(gt_pose[:, 1].max() + y_margin, gt_pose[:, 1].min() - y_margin)
            ax2.set_zlim(gt_pose[:, 2].min() - z_margin, gt_pose[:, 2].max() + z_margin)
            ax2.legend()
            ax2.grid(True)
            image_filename2 = osp.join(osp.expanduser("/home/hyx/test/BCML/SAVE"), f'{self.name}-EEG_only_3d.png')
            fig2.tight_layout()
            fig2.savefig(image_filename2, dpi=300)
            print(f"3D 图像保存至: {image_filename2}")

            fig1, ax1 = plt.subplots()
            ax1.plot(gt_pose[:, 0], gt_pose[:, 2], color='black', marker='o', linestyle='', label='Ground Truth Pose')
            ax1.plot(realeg_pose[:, 0], realeg_pose[:, 2], color='red', marker='s', linestyle='', label='Predicted Pose (predeg_poses)')
            

            for i in range(len(gt_pose)):
                ax1.plot([gt_pose[i, 0], realeg_pose[i, 0]], [gt_pose[i, 2], realeg_pose[i, 2]], color='gray', linestyle='--', alpha=0.5)


            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('Z [m]')
            ax1.legend()
            ax1.grid(True)
    
            image_filename1 = osp.join(osp.expanduser("/home/hyx/test/BCML/SAVE"), f'{self.name}-EEG_only.png')
            fig1.savefig(image_filename1)
            print(f"图像保存至: {image_filename1}")
            plt.close(fig1)
            plt.close(fig2)


        return eval_loss_eeg, eeg_t_mean, eeg_q_mean, eeg_t_variance, eeg_t_std,  eeg_q_variance, eeg_q_std, predeg_poses,targ_poses




    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        test_preds = y_pred                       #np.argmax(y_pred, 1)
        test_truth = y_true
        if to_print:
            print("Confusion Matrix (pos/neg) :")
            print(confusion_matrix(test_truth, test_preds))
            print("Classification Report (pos/neg) :")
            print(classification_report(test_truth, test_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
        return #mean_squared_error(test_truth, test_preds)                         accuracy_score(test_truth, test_preds)

    def get_domain_loss(self,):
        if self.train_config.use_sim:
            return 0.0
        # Predicted domain labels
        domain_pred_e = self.model.domain_label_e
        domain_pred_i = self.model.domain_label_i
        # True domain labels
        domain_true_e = to_gpu(torch.LongTensor([0]*domain_pred_e.size(0)))
        domain_true_i = to_gpu(torch.LongTensor([1]*domain_pred_i.size(0)))
        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_e, domain_pred_i), dim=0)
        domain_true = torch.cat((domain_true_e, domain_true_i), dim=0)
        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_sim_loss(self, label):
        loss = self.loss_sim(self.model.representation_common_e, self.model.representation_common_i, get_shuffled(self.model.representation_common_i, label))
        loss += self.loss_sim(self.model.representation_common_i, self.model.representation_common_e, get_shuffled(self.model.representation_common_e, label))
        return loss

    def get_diff_loss(self):
        common_e = self.model.representation_common_e
        common_i = self.model.representation_common_i
        private_e = self.model.representation_private_e
        private_i = self.model.representation_private_i
        # Between private and common
        loss = self.loss_diff(private_e, common_e)
        loss += self.loss_diff(private_i, common_i)
        # Across privates
        loss += self.loss_diff(private_e, private_i)
        return loss

    def get_recon_loss(self, ):
        loss = self.loss_recon(self.model.representation_e_recon, self.model.representation_e_orig)
        loss += self.loss_recon(self.model.representation_i_recon, self.model.representation_i_orig)
        loss = loss/2.0
        return loss
    


