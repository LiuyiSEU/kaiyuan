import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import Dataset
import Model
from torch import optim
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import LambdaLR


def train():
    t_train_path = '/home/liuyi/RFF/RFFdata/ORI_1d/train'
    t_val_path = '/home/liuyi/RFF/RFFdata/ORI_1d/val'
    s_train_path = '/home/liuyi/RFF/RFFdata/t_rff/train/'
    s_val_path = '/home/liuyi/RFF/RFFdata/t_rff/val/'
    model = Model.teacherNet()
    #t_train_dataset = Dataset.MyDataset(t_train_path)
    #t_train_loader = DataLoader(t_train_dataset, batch_size=1, shuffle=False)
    #t_val_dataset = Dataset.MyDataset(t_val_path)
    #t_val_loader = DataLoader(t_val_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load('/home/liuyi/RFF/res/teacher_model.pth'))
    fc_weights = model.fc.state_dict()
    torch.save(fc_weights, '/home/liuyi/RFF/res/fc_weights.pth')
    '''
    model.eval()
    model.to(torch.double)
    for idx, (t_data, t_labels) in enumerate(t_train_loader):
        t_data = t_data.to(device)
        output_t, t_feature, t1, t2, t3, t4 = model(t_data)
        t_feature_np = t_feature[0].cpu().detach().numpy()
        np.save(s_train_path + str(idx) + '.npy', t_feature_np)
    for idx, (t_data, t_labels) in enumerate(t_val_loader):
        t_data = t_data.to(device)
        output_t, t_feature, t1, t2, t3, t4 = model(t_data)
        t_feature_np = t_feature[0].cpu().detach().numpy()
        np.save(s_val_path + str(idx) + '.npy', t_feature_np)
    '''
