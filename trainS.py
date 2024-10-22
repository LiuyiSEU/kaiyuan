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


def lr_lambda(epoch, warmup_epochs=5, decay_epochs=40):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        decay_phase = (epoch - warmup_epochs) // decay_epochs
        return 0.1 ** decay_phase  # 之后每个 20 个 epoch 乘以 0.1


def train():
    t_train_path = '/home/liuyi/RFF/RFFdata/ORI_1d/train'
    t_val_path = '/home/liuyi/RFF/RFFdata/ORI_1d/val'
    s_train_path = '/home/liuyi/RFF/RFFdata/REC_no/train'
    s_val_path = '/home/liuyi/RFF/RFFdata/REC_no/val'
    num_epoch = 11
    teacher_model = Model.teacherNet()
    student_model = Model.studentNet()
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    t_train_generator = torch.Generator().manual_seed(2024)
    t_val_generator = torch.Generator().manual_seed(2023)
    s_train_generator = torch.Generator().manual_seed(2024)
    s_val_generator = torch.Generator().manual_seed(2023)
    t_train_dataset = Dataset.MyDataset(t_train_path)
    t_train_loader = DataLoader(t_train_dataset, batch_size=32, shuffle=True, generator=t_train_generator)
    t_val_dataset = Dataset.MyDataset(t_val_path)
    t_val_loader = DataLoader(t_val_dataset, batch_size=32, shuffle=True, generator=t_val_generator)
    s_train_dataset = Dataset.MyDataset(s_train_path)
    s_train_loader = DataLoader(s_train_dataset, batch_size=32, shuffle=True, generator=s_train_generator)
    s_val_dataset = Dataset.MyDataset(s_val_path)
    s_val_loader = DataLoader(s_val_dataset, batch_size=32, shuffle=True, generator=s_val_generator)
    cnt = 0
    train_loss = []
    val_loss = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    #student_model.load_state_dict(torch.load('/home/liuyi/RFF/res/teacher_model.pth'))
    student_model.train()
    teacher_model.to(device)
    teacher_model.load_state_dict(torch.load('/home/liuyi/RFF/res/teacher_model.pth'))
    teacher_model.eval()
    teacher_model.to(torch.double)
    student_model.to(torch.double)

    for name, p in teacher_model.named_parameters():
        p.requires_grad = False
    for epoch in range(num_epoch):
        train_epoch_loss = []
        scheduler.step()
        for (t_data, t_labels), (s_data, s_labels) in zip(t_train_loader, s_train_loader):
            t_data = t_data.to(device)
            s_data = s_data.to(device)
            s_feature = student_model(s_data)
            output_t, t_feature = teacher_model(t_data)
            optimizer.zero_grad()
            #t_feature = t_feature * 10
            loss = criterion(s_feature,t_feature)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
        train_loss.append(np.average(train_epoch_loss))

        val_epoch_loss = []
        with torch.no_grad():
            for (t_data, t_labels), (s_data, s_labels) in zip(t_val_loader, s_val_loader):
                t_data = t_data.to(device)
                s_data = s_data.to(device)
                s_labels = s_labels.to(device)
                s_feature = student_model(s_data)
                output_t, t_feature = teacher_model(t_data)
                #t_feature = t_feature * 10
                loss = criterion(s_feature,t_feature)
                val_epoch_loss.append(loss.item())
                if epoch == cnt:
                    print(t_feature.size())
                    cnt = cnt + 5
                    t_np = t_feature[0].squeeze(0).cpu().detach().numpy()
                    s_np = s_feature[0].squeeze(0).cpu().detach().numpy()
                    plt.figure(figsize=(30, 4))
                    plt.plot(t_np, '-o', label="t")
                    plt.plot(s_np, '-o', label="s")
                    plt.title("feature")
                    plt.legend()
                    plt.savefig('/home/liuyi/RFF/res/samples/epoch'+str(epoch)+'.png')
                    torch.save(student_model.state_dict(), '/home/liuyi/RFF/res/samples/student_hint'+str(epoch)+'.pth')
        val_loss.append(np.average(val_epoch_loss))
        print("epoch = {}/{}, train_loss = {}, val_loss = {}".format(epoch + 1, num_epoch, train_loss[-1], val_loss[-1]))

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, '-o', label="train_loss")
    plt.plot(val_loss, '-o', label="val_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.savefig('/home/liuyi/RFF/res/s_loss.png')
    #torch.save(student_model.state_dict(), '/home/liuyi/RFF/res/student_hint.pth')
