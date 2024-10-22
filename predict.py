from matplotlib import pyplot as plt
import torch
import Model
import numpy as np
import Dataset
from torch.utils.data import DataLoader


def model_predict(student_path, data_path, num_samples):
    val_dataset = Dataset.MyDataset(data_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = Model.studentNet()
    student_model.to(device)
    student_model.to(torch.double)
    student_model.load_state_dict(torch.load(student_path))
    student_model.eval()
    epoch_acc = 0.0
    feature = []
    conf_matrix = np.zeros((num_samples, num_samples), dtype=float)
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, fs = student_model(inputs)
            feature.append(fs.cpu().numpy())
            predict_y = torch.max(outputs, dim=1)[1]
            epoch_acc += torch.eq(predict_y, labels).sum().item()
            predict_y_np = predict_y.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i in range(len(predict_y_np)):
                conf_matrix[labels_np[i], predict_y_np[i]] += 1
    epoch_acc = epoch_acc / len(val_dataset)
    print("acc = {}".format(epoch_acc))
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
    plt.title('Confusion Matrix')
    #plt.colorbar(cax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('/home/liuyi/RFF/res/predict_res.png')
    return epoch_acc




