import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import Dataset
import Model,testmodel
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
    s_train_path = '/home/liuyi/RFF/RFFdata/REC_1d/train'
    s_val_path = '/home/liuyi/RFF/RFFdata/REC_1d/val'
    temperature = 7
    alpha = 0
    num_samples = 25
    teacher_model = Model.teacherNet()
    student_model = Model.studentNet()
    extract_model = Model.extractNet()
    optimizer = optim.Adam(extract_model.parameters(), lr=0.0001)
    #optimizer = optim.SGD(extract_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda)

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

    num_epoch = 10
    train_loss = []
    val_loss = []
    acc = []
    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    #student_model.load_state_dict(torch.load('/home/liuyi/RFF/res/student_hint.pth'))
    student_model.eval()
    teacher_model.to(device)
    teacher_model.load_state_dict(torch.load('/home/liuyi/RFF/res/teacher_model.pth'))
    teacher_model.eval()
    extract_model.to(device)
    #extract_model.load_state_dict(torch.load('/home/liuyi/RFF/res/extract_model.pth'))
    extract_model.train()
    teacher_model.to(torch.double)
    student_model.to(torch.double)
    extract_model.to(torch.double)

    for name, p in teacher_model.named_parameters():
        p.requires_grad = False
    for name, p in student_model.named_parameters():
        p.requires_grad = False
    for epoch in range(num_epoch):
        
        train_epoch_loss = []
        scheduler.step()
        for (t_data, t_labels), (s_data, s_labels) in zip(t_train_loader, s_train_loader):
            t_data = t_data.to(device)
            s_data = s_data.to(device)
            s_labels = s_labels.to(device)
            #output_s = student_model(s_data)
            output_t, t_feature = teacher_model(t_data)
            output_e = extract_model(t_feature)
            optimizer.zero_grad()
            loss = criterion(output_e, s_labels)
            #loss = Model.distillation_loss(output_t, output_e, s_labels, temperature, alpha)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
        train_loss.append(np.average(train_epoch_loss))
        
        val_epoch_loss = []
        epoch_acc = 0.0
        conf_matrix = np.zeros((num_samples, num_samples), dtype=float)
        with torch.no_grad():
            for (t_data, t_labels), (s_data, s_labels) in zip(t_val_loader, s_val_loader):
                t_data = t_data.to(device)
                s_data = s_data.to(device)
                s_labels = s_labels.to(device)
                #output_s = student_model(s_data)
                output_t, t_feature = teacher_model(t_data)
                output_e = extract_model(t_feature)
                predict_y = torch.max(output_e, dim=1)[1]
                loss = criterion(output_e, s_labels)
                #loss = Model.distillation_loss(output_t, output_e, s_labels, temperature, alpha)
                epoch_acc += torch.eq(predict_y, s_labels).sum().item()
                val_epoch_loss.append(loss.item())
                predict_y_np = predict_y.cpu().numpy()
                labels_np = s_labels.cpu().numpy()
                for i in range(len(predict_y_np)):
                    conf_matrix[labels_np[i], predict_y_np[i]] += 1
        epoch_acc = epoch_acc / len(s_val_dataset)
        val_loss.append(np.average(val_epoch_loss))
        acc.append(epoch_acc)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(extract_model.state_dict(), '/home/liuyi/RFF/res/extract_model.pth')
            row_sums = conf_matrix.sum(axis=1)
            conf_matrix_prob = conf_matrix / row_sums[:, np.newaxis]
            res_matrix = (conf_matrix_prob * 100).astype(int)
            savepath = '/home/liuyi/RFF/res/Confusion_Matrix.npy'
            np.save(savepath, res_matrix)
            fig, ax = plt.subplots(figsize=(16, 16))
            cax = ax.matshow(res_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=res_matrix.max())
            for i in range(res_matrix.shape[0]):
                for j in range(res_matrix.shape[1]):
                    text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
                    plt.text(j, i, str(res_matrix[i, j]), ha='center', va='center', color=text_color)
            label_list = [str(label) for label in range(res_matrix.shape[1])]
            plt.xticks(np.arange(res_matrix.shape[1]), label_list)
            plt.yticks(np.arange(res_matrix.shape[0]), label_list)
            plt.title('Confusion Matrix, Best Acc = '+str(best_acc))
            #plt.colorbar(cax)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig('/home/liuyi/RFF/res/e_res.png')
        print("epoch = {}/{}, val_loss = {}, acc = {}".format(epoch + 1, num_epoch, val_loss[-1], epoch_acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss, '-o', label="train_loss")
    plt.plot(val_loss, '-o', label="val_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(acc)
    plt.title("Acc")
    plt.savefig('/home/liuyi/RFF/res/e_loss.png')

