# Imports
import numpy as np
import torch
import torch.nn as nn
from utils import ReverseLayerF
import torch.nn.functional as F
import math
import torchvision
from torchvision import  models      
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd
import torch.nn.init as init
from torch import Tensor
from torch.autograd import Variable
from einops.layers.torch import Rearrange

class EEGNet(nn.Module):
    def __init__(self,T,C):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25
        self.C = C
        self.T = T
        #500
        self.weights = nn.Parameter(torch.Tensor(1, self.C, self.T))
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
    def reset_parameters(self):
        stdv = 1. / math.sqrt(1)
        self.weights.data.uniform_(-stdv, stdv)
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        feature_map = self.block_3(x)  
        x = feature_map.view(feature_map.size(0), -1)  
        return x
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)
        self.linear = nn.Linear(in_features=256 * 4 * 4, out_features=1000)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        # x = self.linear(x)
        return x
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  
# BMCL model
class BMCL(nn.Module):
    def __init__(self, config):
        super(BMCL, self).__init__()

        self.config = config
        self.EEG_size = config.embedding_size           #embedding_size=300
        self.image_size = config.embedding_size

        self.input_sizes = input_sizes = [self.EEG_size, self.image_size]
        self.hidden_sizes = hidden_sizes = [int(self.EEG_size), int(self.image_size)]           #hidden_sizes=128
        self.output_size = output_size = config.num_classes                #class_opt = 7
        self.dropout_rate = dropout_rate = config.dropout                  #0.5
        self.activation = self.config.activation()                                      #relu
        self.tanh = nn.Tanh()
        
        # EEG model choices: EEGChannelNet(self.config) torch.load(self.config.pretrained_net) Model() 
        # Image model choices: ResNet18() ResNet50() AlexNet() ResNet18()

        
        if self.config.data == 'facial':
            #Enc_eeg()
            T=100
            C=60
            self.eeg_model =EEGNet(T,C)
            feature_extractor = models.resnet34(pretrained=True)
            self.image_model = Image(feature_extractor, droprate=0.5, pretrained=True, lstm=False)
            self.eeg_size = T//100*48
            #T//100*48
            #16*15
            self.image_size = 256 * 4 * 4

        for param in self.eeg_model.parameters():
            param.requires_grad = self.config.requires_grad
        for param in self.image_model.parameters():
            param.requires_grad = self.config.requires_grad

        # mapping modalities to same sized space
        self.project_e = nn.Sequential()
        self.project_e.add_module('project_e', nn.Linear(in_features=self.eeg_size, out_features=config.hidden_size))
        self.project_e.add_module('project_e_activation', self.activation)
        self.project_e.add_module('project_e_layer_norm', nn.BatchNorm1d(config.hidden_size))
        self.project_i = nn.Sequential()
        self.project_i.add_module('project_i', nn.Linear(in_features=self.image_size, out_features=config.hidden_size))
        self.project_i.add_module('project_i_activation', self.activation)
        self.project_i.add_module('project_i_layer_norm', nn.BatchNorm1d(config.hidden_size))

        # private encoders
        self.private_e = nn.Sequential()
        self.private_e.add_module('private_e_1', nn.Linear(in_features=config.hidden_size, out_features=64))
        self.private_e.add_module('private_e_activation_1', self.activation)
        self.private_i = nn.Sequential()
        self.private_i.add_module('private_i_1', nn.Linear(in_features=config.hidden_size, out_features=64))
        self.private_i.add_module('private_i_activation_1', self.activation)
        
        # common encoder
        self.common = nn.Sequential()
        self.common.add_module('common_1', nn.Linear(in_features=config.hidden_size, out_features=64))
        self.common.add_module('common_activation_1', self.activation)


        self.private_e_2 = nn.Sequential()
        self.private_e_2.add_module('private_e_2', nn.Linear(in_features=config.hidden_size, out_features=32))
        self.private_e_2.add_module('private_e_activation_2', self.activation)
        self.private_i_2 = nn.Sequential()
        self.private_i_2.add_module('private_i_2', nn.Linear(in_features=config.hidden_size, out_features=32))
        self.private_i_2.add_module('private_i_activation_2', self.activation)
        
        # common encoder
        self.common_2 = nn.Sequential()
        self.common_2.add_module('common_2', nn.Linear(in_features=config.hidden_size, out_features=32))
        self.common_2.add_module('common_activation_2', self.activation)




        self.private_e_2 = nn.Sequential()
        self.private_e_2.add_module('private_e_2', nn.Linear(in_features=config.hidden_size, out_features=32))
        self.private_e_2.add_module('private_e_activation_2', self.activation)
        self.private_i_2 = nn.Sequential()
        self.private_i_2.add_module('private_i_2', nn.Linear(in_features=config.hidden_size, out_features=32))
        self.private_i_2.add_module('private_i_activation_2', self.activation)
        
        # common encoder
        self.common_2 = nn.Sequential()
        self.common_2.add_module('common_2', nn.Linear(in_features=config.hidden_size, out_features=32))
        self.common_2.add_module('common_activation_2', self.activation)


        self.private_e_3 = nn.Sequential()
        self.private_e_3.add_module('private_e_3', nn.Linear(in_features=64, out_features=16))
        self.private_e_3.add_module('private_e_activation_3', self.activation)
        self.private_i_3 = nn.Sequential()
        self.private_i_3.add_module('private_i_3', nn.Linear(in_features=64, out_features=16))
        self.private_i_3.add_module('private_i_activation_3', self.activation)
        
        # common encoder
        self.common_3 = nn.Sequential()
        self.common_3.add_module('common_3', nn.Linear(in_features=64, out_features=16))
        self.common_3.add_module('common_activation_3', self.activation)





        # # reconstruct
        # self.recon_e = nn.Sequential()
        # self.recon_e.add_module('recon_e_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.recon_i = nn.Sequential()
        # self.recon_i.add_module('recon_i_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        # if not self.config.use_sim:
        #     self.discriminator = nn.Sequential()
        #     self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        #     self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
        #     self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
        #     self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        # # common-private collaborative discriminator
        # self.sp_discriminator = nn.Sequential()
        # self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))

        # fusion
        # self.fusion_e_xyz = nn.Sequential()
        # self.fusion_e_xyz.add_module('fusion_layer_1', nn.Linear(in_features=32, out_features=32))
        # self.fusion_e_xyz.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        # self.fusion_e_xyz.add_module('fusion_layer_1_activation', self.activation)
        # self.fusion_e_xyz.add_module('fusion_layer_3', nn.Linear(in_features=32, out_features=6))  # 回归位置 (x, y, z)   #3


        # self.fusion_i_xyz = nn.Sequential()
        # self.fusion_i_xyz.add_module('fusion_layer_1', nn.Linear(in_features=32, out_features=32))
        # self.fusion_i_xyz.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        # self.fusion_i_xyz.add_module('fusion_layer_1_activation', self.activation)
        # self.fusion_i_xyz.add_module('fusion_layer_3', nn.Linear(in_features=32, out_features=6))  # 回归位置 (x, y, z)

            # fusion
        self.fusion_e_xyz = nn.Sequential()
        self.fusion_e_xyz.add_module('fusion_layer_1', nn.Linear(in_features=64, out_features=64))
        self.fusion_e_xyz.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion_e_xyz.add_module('fusion_layer_1_activation', self.activation)
        self.fusion_e_xyz.add_module('fusion_layer_3', nn.Linear(in_features=64, out_features=6))  # 回归位置 (x, y, z)   #3


        self.fusion_i_xyz = nn.Sequential()
        self.fusion_i_xyz.add_module('fusion_layer_1', nn.Linear(in_features=64, out_features=64))
        self.fusion_i_xyz.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion_i_xyz.add_module('fusion_layer_1_activation', self.activation)
        self.fusion_i_xyz.add_module('fusion_layer_3', nn.Linear(in_features=64, out_features=6))  # 回归位置 (x, y, z)


    def alignment(self, eeg, image):


        # extract features from EEG modality
        representation_eeg = self.eeg_model(eeg)
        representation_image = self.image_model(image)

        # common-private encoders
        self.common_private(representation_eeg, representation_image)

        # if not self.config.use_sim:
        #     # discriminator
        #     reversed_common_code_e = ReverseLayerF.apply(self.representation_common_e, self.config.reverse_grad_weight)
        #     reversed_common_code_i = ReverseLayerF.apply(self.representation_common_i, self.config.reverse_grad_weight)
        #     self.domain_label_e = self.discriminator(reversed_common_code_e)
        #     self.domain_label_i = self.discriminator(reversed_common_code_i)
        # else:
        #     self.domain_label_e = None
        #     self.domain_label_i = None

        # self.common_or_private_p_e = self.sp_discriminator(self.representation_private_e)
        # self.common_or_private_p_i = self.sp_discriminator(self.representation_private_i)
        # self.common_or_private_s = self.sp_discriminator( (self.representation_common_e + self.representation_common_i)/2.0 )
        
        # For reconstruction
        # self.reconstruct()
        
        h_e = torch.stack((self.representation_private_e, self.representation_common_e), dim=0)
        h_e = torch.cat((h_e[0], h_e[1]), dim=1)
        o_e=self.fusion_e_xyz(h_e)
        h_i = torch.stack((self.representation_private_i, self.representation_common_i), dim=0)
        h_i = torch.cat((h_i[0], h_i[1]), dim=1)
        o_i=self.fusion_i_xyz(h_i)
        return o_e, o_i
    
    def reconstruct(self,):
        self.representation_e = (self.representation_private_e + self.representation_common_e)
        self.representation_i = (self.representation_private_i + self.representation_common_i)

        self.representation_e_recon = self.recon_e(self.representation_e)
        self.representation_i_recon = self.recon_i(self.representation_i)

    def common_private(self, representation_e, representation_i):
        # Projecting to same sized space
        self.representation_e_orig = representation_e=  self.project_e(representation_e)
        self.representation_i_orig = representation_i = self.project_i(representation_i)           #特征映射同一层
        # Private-common components
        self.representation_private_e = self.private_e(representation_e)                    
        self.representation_private_i = self.private_i(representation_i)
        self.representation_common_e = self.common(representation_e)
        self.representation_common_i = self.common(representation_i)
        
        self.rep_e=torch.stack((self.representation_private_e, self.representation_common_e), dim=0)
        self.rep_e=torch.cat((self.rep_e[0], self.rep_e[1]), dim=1)
        self.rep_i=torch.stack((self.representation_private_i, self.representation_common_i), dim=0)
        self.rep_i=torch.cat((self.rep_i[0], self.rep_i[1]), dim=1)
        self.representation_private_e = self.private_e_2(self.rep_e)    # 
        self.representation_private_i = self.private_i_2(self.rep_i)    # 
        self.representation_common_e = self.common_2(self.rep_e)         # 
        self.representation_common_i = self.common_2(self.rep_i)
        
        # self.rep_e=torch.stack((self.representation_private_e, self.representation_common_e), dim=0)
        # self.rep_e=torch.cat((self.rep_e[0], self.rep_e[1]), dim=1)
        # self.rep_i=torch.stack((self.representation_private_i, self.representation_common_i), dim=0)
        # self.rep_i=torch.cat((self.rep_i[0], self.rep_i[1]), dim=1)
        # self.representation_private_e = self.private_e_3(self.rep_e)    # 
        # self.representation_private_i = self.private_i_3(self.rep_i)    # 
        # self.representation_common_e = self.common_3(self.rep_e)         # 
        # self.representation_common_i = self.common_3(self.rep_i)
        
        
        
        
    def forward(self, eeg, image):
        self.eeg = eeg
        self.image = image
        batch_size = eeg.size(0)
        o_e, o_i = self.alignment(eeg, image)
        return o_e, o_i

class Image(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=4096, lstm=False):
        super(Image, self).__init__()
        self.droprate = droprate
        self.lstm = lstm
        feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)                #修改平均池化层其输出大小为(1, 1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)             #新全连接层的输入特征数设置为fe_out_planes（即原始全连接层的输入特征数），
                                                                                    #输出特征数设置为feat_dim
        self.att = AttentionBlock(feat_dim)               #注意力块（AttentionBlock）
        self.fc_xyz = nn.Linear(feat_dim, 3)                #全连接层（nn.Linear），用于将feat_dim维的特征向量映射到3维空间
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):          #检查它是否是 nn.Conv2d（二维卷积层）或 nn.Linear（全连接层）的实例
                nn.init.kaiming_normal_(m.weight.data)                    #则使用 Kaiming 初始化（也称为 He 初始化）来初始化权重，并将偏置（如果有的话）初始化为 0。
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        x = self.feature_extractor(x)                  #特征提取器（self.feature_extractor）处理输入 x，然后应用 ReLU 激活函数。
        x = F.relu(x)
        if self.lstm:
            x = self.lstm4dir(x)
        else:
            x = self.att(x.view(x.size(0), -1))
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate) 
        return x            #返回特征

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):               #inchannel ==2048
        super(AttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        g_x = self.g(x).view(batch_size, out_channels // 8, 1)                #被重新塑形（reshape）为一个三维张量，其维度分别为 batch_size、out_channels // 8 和 1。

        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)                                          #将 theta_x 的维度重新排列为 (batch_size, 1, out_channels // 8)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)

        f = torch.matmul(phi_x, theta_x)                      #进行矩阵乘法
        f_div_C = F.softmax(f, dim=-1)                           #可以将一个向量压缩到另一个向量中，使得每一个元素的范围都在 (0, 1) 之间，并且所有元素的和为 1

        y = torch.matmul(f_div_C, g_x)

        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z
   

