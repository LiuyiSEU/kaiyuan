from sklearn.model_selection import RandomizedSearchCV
import torch
import torch.optim as optim
import torch.nn as nn
import testmodel
from torch.utils.data import DataLoader
import Dataset



train_dataset_path = r'/home/liuyi/RFF/RFFdata/LoRa_mini/train'
val_dataset_path = r'/home/liuyi/RFF/RFFdata/LoRa_mini/val'
num_samples = 10

train_dataset = Dataset.MyDataset(train_dataset_path, num_samples)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataset = Dataset.MyDataset(val_dataset_path, num_samples)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

# 定义深度学习模型
model = testmodel.ResNet(testmodel.Bottleneck, [3, 4, 6, 3], num_samples)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# 定义参数空间
param_dist = {
    'lr': [0.0001, 0.001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009],
    'weight_decay': [0.0001, 0.001, 0.01, 0.1]
}

# 创建随机搜索对象
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10,
                                   scoring='accuracy', cv=3, verbose=2, random_state=42, n_jobs=-1)

# 执行随机搜索
random_search.fit(train_loader)

# 输出最佳参数组合和对应的准确率
print("Best parameters found: ", random_search.best_params_)
print("Best accuracy: {:.2f}".format(random_search.best_score_))

# 使用最佳参数组合的模型在测试集上进行评估
best_model = random_search.best_estimator_
best_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print("Test accuracy with best parameters: {:.2f}%".format(accuracy))
