import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import Model,testmodel
import Dataset
import matplotlib.pyplot as plt


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)


def lr_lambda(epoch, warmup_epochs=5, decay_epochs=20):
    if epoch < warmup_epochs:
        #return 0.1
        return (epoch + 1) / warmup_epochs  # 在前 warmup_epochs 个 epoch 内线性增加
    else:
        # 计算当前处于第几个 20-epoch 阶段
        decay_phase = (epoch - warmup_epochs) // decay_epochs
        #return 1
        return 0.1 ** decay_phase  # 之后每个 20 个 epoch 乘以 0.1


def train():
    train_dataset_path = r'/home/liuyi/ORI_1/train'
    val_dataset_path = r'/home/liuyi/ORI_1/val'
    num_samples = 25

    train_dataset = Dataset.MyDataset(train_dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = Dataset.MyDataset(val_dataset_path)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    #model = testmodel.ResNet(testmodel.Bottleneck, [3, 4, 6, 3], num_samples)
    model = Model.teacherNet()
    #model = Model.ResNet(Model.BasicBlock, [3, 4, 6, 3], num_classes=5, include_top=True)
    #model = Model.Resnet(layers=[3, 4, 6, 3], data_channels=1, num_classes=num_samples)
    #model.apply(weight_init)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0007)
    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()


    num_epoch = 40
    b = 0.5
    train_loss = []
    val_loss = []
    acc = []
    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.load_state_dict(torch.load('/home/liuyi/RFF/res/teacher_model.pth'))
    model.to(device)
    for epoch in range(num_epoch):
        scheduler.step()
        '''
        if epoch < 5:
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
        '''
        if epoch > 10:
            b = 0.4
        elif epoch > 15:
            b = 0.2
        elif epoch > 20:
            b = 0.1
        elif epoch > 25:
            b = 0
        model.train()
        model = model.to(torch.double)
        train_epoch_loss = []
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            #input_data = torch.cat([inputs.real.unsqueeze(1), inputs.imag.unsqueeze(1)], dim=1)
            labels = labels.to(device)
            outputs, f = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            flood = (loss-b).abs()+b
            loss.backward()
            optimizer.step()
            #scheduler.step()
            train_epoch_loss.append(loss.item())
        train_loss.append(np.average(train_epoch_loss))

        model.eval()
        val_epoch_loss = []
        epoch_acc = 0.0
        conf_matrix = np.zeros((num_samples, num_samples), dtype=float)
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                #input_data = torch.cat([inputs.real.unsqueeze(1), inputs.imag.unsqueeze(1)], dim=1)
                labels = labels.to(device)
                outputs,f = model(inputs)
                predict_y = torch.max(outputs, dim=1)[1]
                epoch_acc += torch.eq(predict_y, labels).sum().item()
                loss = criterion(outputs, labels)
                val_epoch_loss.append(loss.item())
                predict_y_np = predict_y.cpu().numpy()
                labels_np = labels.cpu().numpy()
                for i in range(len(predict_y_np)):
                    conf_matrix[labels_np[i], predict_y_np[i]] += 1
        epoch_acc = epoch_acc / len(val_dataset)
        val_loss.append(np.average(val_epoch_loss))
        acc.append(epoch_acc)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), '/home/liuyi/RFF/res/teacher_model.pth')
            row_sums = conf_matrix.sum(axis=1)
            conf_matrix_prob = conf_matrix / row_sums[:, np.newaxis]
            res_matrix = (conf_matrix_prob * 100).astype(int)
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
            plt.savefig('/home/liuyi/RFF/res/t_res.png')
            np.save('/home/liuyi/RFF/res/t_res.npy', res_matrix)
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
    #plt.show()
    plt.savefig('/home/liuyi/RFF/res/t_loss.png')
    #torch.save(model.state_dict(), '/home/liuyi/RFF/res/teacher_model.pth')


def predict(val):
    val = torch.from_numpy(val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val.to(device)
    val = val.unsqueeze(0)
    input_data = torch.cat([val.real.unsqueeze(1), val.imag.unsqueeze(1)], dim=1)
    model = Model.teacherNet()
    model.to(torch.double)
    model.load_state_dict(torch.load('teacher_model.pth'))
    model.eval()
    res = model(input_data)
    predict_y = torch.max(res, dim=1)[1]
    res = predict_y.detach().numpy()
    print("识别结果为: device{}".format(res[0]+1))
