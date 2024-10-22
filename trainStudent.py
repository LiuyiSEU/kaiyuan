import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import Dataset
import Model,testmodel,predict
from torch import optim
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import LambdaLR


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)


def lr_lambda(epoch, warmup_epochs=5, decay_epochs=20):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # 在前 warmup_epochs 个 epoch 内线性增加
    else:
        # 计算当前处于第几个 20-epoch 阶段
        decay_phase = (epoch - warmup_epochs) // decay_epochs
        #return 1
        return 0.1 ** decay_phase  # 之后每个 20 个 epoch 乘以 0.1


def train():
    t_train_path = '/home/liuyi/ORI_1/train'
    t_val_path = '/home/liuyi/ORI_1/val'
    s_train_path = '/home/liuyi/REC_1/train'
    s_val_path = '/home/liuyi/REC_1/val'
    temperature = 7
    alpha = 0.3
    x = 0.9
    use_hint=False
    num_samples = 25
    teacher_model = Model.teacherNet()
    cfgs = {
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }
    #student_model = testmodel.VGG16(5)
    #student_model = testmodel.VGG(testmodel.make_layers(cfgs["VGG19"]))
    #student_model = testmodel.ResNet(testmodel.Bottleneck,[3,4,6,3],25)
    student_model = Model.studentNet()
    student_model.apply(weight_init)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda)
    criterionA = nn.MSELoss()
    criterionB = nn.CrossEntropyLoss()

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

    num_epoch = 50
    train_loss = []
    val_loss = []
    acc = []
    pre_acc = []
    best_acc = 0.0
    best_pre = 0.0
    best_epoch = 0
    b = 4.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    #if not use_hint:
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
            s_labels = s_labels.to(device)
            output_s, f_s = student_model(s_data)
            output_t, f_t = teacher_model(t_data)
            optimizer.zero_grad()
            if use_hint:
                #loss = criterionA(f_s, f_t)+0.2*criterionA(s1,t1)+0.2*criterionA(s2,t2)+0.2*criterionA(s3,t3)+0.2*criterionA(s4,t4)
                loss = criterionA(f_s, f_t)
            else:
                loss = 0.5 * criterionA(f_s, f_t) + Model.distillation_loss(output_t, output_s, s_labels, temperature, alpha)
                
            loss.backward()
            optimizer.step()
            #scheduler.step()
            train_epoch_loss.append(loss.item())
        train_loss.append(np.average(train_epoch_loss))
        '''
        for name, param in student_model.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name}, Gradient Norm: {torch.norm(param.grad)}")
            else:
                print(f"Layer: {name}, Gradient: None")
        '''
        val_epoch_loss = []
        epoch_acc = 0.0
        conf_matrix = np.zeros((num_samples, num_samples), dtype=float)
        with torch.no_grad():
            for (t_data, t_labels), (s_data, s_labels) in zip(t_val_loader, s_val_loader):
                t_data = t_data.to(device)
                #t_input_data = torch.cat([t_data.real.unsqueeze(1), t_data.imag.unsqueeze(1)], dim=1)
                s_data = s_data.to(device)
                #s_input_data = torch.cat([s_data.real.unsqueeze(1), s_data.imag.unsqueeze(1)], dim=1)
                s_labels = s_labels.to(device)
                output_s,f_s = student_model(s_data)
                output_t,f_t = teacher_model(t_data)
                #output_s,s_hint = student_model(s_data)
                #output_t,t_hint = teacher_model(t_data)
                if use_hint:
                    #loss = criterionA(f_s, f_t)+0.2*criterionA(s1,t1)+0.2*criterionA(s2,t2)+0.2*criterionA(s3,t3)+0.2*criterionA(s4,t4)
                    loss = criterionA(f_s, f_t)
                else:
                    predict_y = torch.max(output_s, dim=1)[1]
                    loss = 0.5 *  criterionA(f_s, f_t) + Model.distillation_loss(output_t, output_s, s_labels, temperature, alpha)
                    #loss = criterionB(output_s, s_labels)
                    #print(criterionA(f_s,f_t))
                    #print(criterionB(output_s, s_labels))
                    epoch_acc += torch.eq(predict_y, s_labels).sum().item()
                #predict_y = torch.max(output_s, dim=1)[1]
                #loss = Model.hint_loss(s_hint,t_hint)+ 0.5 * Model.distillation_loss(output_t, output_s, s_labels, temperature, alpha)
                #epoch_acc += torch.eq(predict_y, s_labels).sum().item()
                val_epoch_loss.append(loss.item())
                if not use_hint:
                    predict_y_np = predict_y.cpu().numpy()
                    labels_np = s_labels.cpu().numpy()
                    for i in range(len(predict_y_np)):
                        conf_matrix[labels_np[i], predict_y_np[i]] += 1
        epoch_acc = epoch_acc / len(s_val_dataset)
        if not best_acc == 0.0:
            model_path = '/home/liuyi/RFF/res/student_model.pth'
            data_path = '/home/liuyi/test_1d'
            predict_acc = predict.model_predict(model_path, data_path, 25)
            pre_acc.append(predict_acc)
            print("predict_acc=".format(predict_acc))
            if predict_acc > best_pre:
                best_pre = predict_acc
                torch.save(student_model.state_dict(), '/home/liuyi/RFF/res/best_pre.pth')
                best_epoch = epoch
        val_loss.append(np.average(val_epoch_loss))
        acc.append(epoch_acc)
        if epoch_acc > best_acc and not use_hint:
            best_acc = epoch_acc
            torch.save(student_model.state_dict(), '/home/liuyi/RFF/res/student_model.pth')
            #torch.save(student_model.state_dict(), '/home/liuyi/RFF/res/student_model.pth')
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
            #plt.:colorbar(cax)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig('/home/liuyi/RFF/res/s_res.png')
            np.save('/home/liuyi/RFF/res/REC_1.npy',conf_matrix)
        print("epoch = {}/{}, val_loss = {}, acc = {}".format(epoch + 1, num_epoch, val_loss[-1], epoch_acc))

    print("best epoch = {}, predict Acc = {}".format(best_epoch, best_pre))
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss, '-o', label="train_loss")
    plt.plot(val_loss, '-o', label="val_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(acc,'-o', label="train_acc")
    plt.plot(pre_acc,'-o', label="pre_acc")
    plt.title("Acc")
    plt.savefig('/home/liuyi/RFF/res/s_loss.png')
    if use_hint:
        torch.save(student_model.state_dict(), '/home/liuyi/RFF/res/student_hint.pth')

'''
def predict(val):
    val = torch.from_numpy(val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = val.unsqueeze(0)
    input_data = torch.cat([val.real.unsqueeze(1), val.imag.unsqueeze(1)], dim=1)
    input_data.to(device)
    model = Model.studentNet()
    model.to(torch.double)
    model.load_state_dict(torch.load('student_model.pth', map_location=torch.device('cpu')))
    model.cpu()
    model.eval()
    res = model(input_data)
    predict_y = torch.max(res, dim=1)[1]
    res = predict_y.detach().numpy()
    print("识别结果为: device{}".format(res[0] + 1))
'''

def extract(train_path):
    train_dataset = Dataset.MyDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    print(train_dataset.__len__())
    model = Model.teacherNet(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch.double)
    model.load_state_dict(torch.load('/home/liuyi/RFF/res/teacher_model.pth'))
    model.to(device)
    model.eval()
    cnt = 1
    print('Start Extract')
    for idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.cpu().numpy()
        output = model(data)
        savepath = '/home/liuyi/RFF/RFFdata/s_rff/val/device' + str(label[0] + 1) + '/' + str(cnt) + '.npy'
        cnt = cnt + 1
        output = output.cpu().detach().numpy()
        np.save(savepath, output)
    print('Extract End')
