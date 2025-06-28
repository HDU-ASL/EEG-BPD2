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
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from scipy.io import savemat
import numpy as np
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
    
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(model, self.train_config.model)(self.train_config)
        # Final list
        for name, param in self.model.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            #print('\t' + name, param.requires_grad)
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        self.param_list = [{'params': self.model.parameters()}]
        if hasattr(self.train_config, 'sax') and hasattr(self.train_config, 'saq'):
            print('learn_beta')
            self.param_list.append({'params': [self.train_criterion.sax, self.train_criterion.saq]})
            
        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                self.param_list,
                #filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

    @time_desc_decorator('Training Start!')
    def eval(self, mode=None, to_print=False):
        assert(mode is not None)
        self.name="Sub5-desk1-NP-100"
        # "/home/hyx/test/BCML/SAVE/Sub1-desk1-NP-100.txt"
        self.model = self.model.to("cpu")
        eval_model = self.model 
        model_params_path = "/home/hyx/test/BCML/models/"+f"{self.name}model_params.pth"
        model_params = torch.load(model_params_path)
        eval_model.project_e.load_state_dict(model_params['project_e'])
        eval_model.private_e.load_state_dict(model_params['private_e'])
        eval_model.common.load_state_dict(model_params['common'])
        eval_model.private_e_2.load_state_dict(model_params['private_e_2'])
        eval_model.common_2.load_state_dict(model_params['common_2'])
        

        eval_model = to_gpu(eval_model)
        assert mode is not None
        eval_model.eval()
        
        for dem in range(6):                       

            if mode == "dev":
                dataloader = self.dev_data_loader
            elif mode == "train":
                dataloader = self.train_data_loader
 
            accumulated_gradients = None
            total_batches = 0 
            
            for eeg, label, image in dataloader:
                total_batches += 1
                y=label
                # y = np.zeros((len(label), 6))
                # for i in range(len(y)):
                #     p = label[i, :3]
                #     q = label[i, 3:7]
                #     q *= np.sign(q[0])  # constrain to hemisphere
                #     back = qlog(q)
                #     y[i] = np.hstack((p, back))
                # y = torch.tensor(y, dtype=torch.float32)  # Ensure y is a tensor of correct type


                eeg.requires_grad_(True)
                eeg.retain_grad()  
                eeg = to_gpu(eeg).detach().requires_grad_()  
                y = to_gpu(y)
                
                representation_eeg = eval_model.eeg_model(eeg)  
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
                eeg.retain_grad()  

                loss = torch.mean((y_tilde_eeg[:, dem] - y[:, dem])**2)

                loss.backward()
                eeg_gradient = eeg.grad
                print(f"eeg gradient shape for batch {total_batches}: {eeg_gradient.shape}")
                channel_gradients = eeg_gradient.mean(dim=3)

                # 累加梯度
                if accumulated_gradients is None:
                    accumulated_gradients = channel_gradients
                else:
                    accumulated_gradients += channel_gradients

            average_gradients = accumulated_gradients / total_batches
            print(f"Average eeg gradient shape: {average_gradients.shape}")
            average_gradients = average_gradients.squeeze()  # 去掉所有大小为 1 的维度
            values=average_gradients.detach().cpu().numpy()
            #归一化
            cam_mean = (values - values.min()) / (values.max() - values.min())
            file_name = f"Sub05{dem}.mat"
            save_path = os.path.join("/home/hyx/test/BCML/salience_maps", file_name)

            # 动态变量名写入 mat 文件
            mat_data = {"cam_mean": cam_mean}
            savemat(save_path, mat_data)
                        
            electrode_matrix = np.array([
        [0, 0, 0, 'FP1', 'FPZ', 'FP2', 0, 0, 0],
        [0, 0, 0, 'AF3', 0, 'AF4', 0, 0, 0],
        ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
        ['FC7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FC8'],
        ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
        ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'CP8'],
        ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P7'],
        ['PO7', 'PO5', 'PO3', 0, 'POZ', 0, 'PO4', 'PO6', 'PO8'],
        [0, 0, 0, 'O1', 'OZ', 'O2', 0, 0, 0]
    ])

            significance_matrix = np.zeros((9, 9))
            current_idx = 0
            for row in range(electrode_matrix.shape[0]):
                for col in range(electrode_matrix.shape[1]):
                        if electrode_matrix[row, col] != "0":  # 非零位置
                            significance_matrix[row, col] = cam_mean[current_idx]
                            current_idx += 1


            colors = ["blue","white", "red"]
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

            plt.figure(figsize=(8, 8))


            significance_matrix_with_nan = np.where(significance_matrix == 0, np.nan, significance_matrix)

            zoom_factor = 1
            high_res_matrix = ndimage.zoom(significance_matrix, zoom_factor, order=3)  

            plt.figure(figsize=(9, 9))

            plt.imshow(high_res_matrix, cmap=cmap, interpolation='bicubic')


            cbar = plt.colorbar(label='Significance Level')
            cbar.set_ticks([0, 0.5, 1])  

            plt.xticks(np.arange(0, 9, 1), fontsize=12)  
            plt.yticks(np.arange(0, 9, 1), fontsize=12) 

            plt.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.5)
            plt.show()
            image_filename = osp.join(osp.expanduser("/home/hyx/test/BCML/salience_maps"), f"{self.name}"+'{0}.png'.format(dem))
            plt.savefig(image_filename, bbox_inches='tight', dpi=300)
            print(image_filename)
            plt.close()













