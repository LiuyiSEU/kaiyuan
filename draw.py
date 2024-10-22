import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

# def ROC():
#     conf_matrix = np.load('./res/D2_15.npy')
#     sums = conf_matrix.sum()
#     colsum = conf_matrix.sum(axis=0)
#     rowsum = conf_matrix.sum(axis=1)
#     Precision = []
#     Recall = []
#     F1_Score = []
#     for i in range(25):
#         TP = conf_matrix[i, i]
#         FN = rowsum[i] - TP
#         FP = colsum[i] - TP
#         TN = sums - TP - FN - FP
#         Precision.append(TP / (TP + FP))
#         Recall.append(TP / (TP + FN))
#         F1_Score.append((2 * Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#     print('Precision = {}, Recall = {}, F1_Score = {}'.format(np.mean(Precision), np.mean(Recall), np.mean(F1_Score)))
#
#
# ROC()


# conf_matrix = np.array([[94, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0], #1
#                         [0, 77, 0, 10, 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0], #2
#                         [0, 0, 84, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 3, 4, 0, 0, 1, 0, 3, 0, 0, 0, 0], #3
#                         [1, 18, 0, 74, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], #4
#                         [0, 8, 0, 4, 71, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, 0, 0, 0], #5
#                         [0, 0, 0, 0, 0, 61, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0], #6
#                         [0, 0, 0, 0, 3, 0, 45, 0, 0, 16, 6, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 3], #7
#                         [0, 0, 0, 0, 0, 1, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 1, 0, 0, 0], #8
#                         [0, 0, 0, 0, 0, 0, 0, 3, 54, 0, 0, 0, 0, 0, 0, 0, 40, 2, 0, 0, 0, 0, 0, 0, 0], #9
#                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 91, 2, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], #10
#                         [0, 0, 0, 0, 7, 0, 3, 0, 0, 4, 55, 0, 0, 0, 0, 0, 0, 0, 10, 18, 0, 0, 0, 0, 0], #11
#                         [0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 0, 64, 8, 0, 10, 1, 0, 0, 0, 0, 2, 0, 0, 0, 9], #12
#                         [0, 0, 1, 0, 0, 0, 1, 0, 0, 10, 1, 10, 59, 0, 6, 0, 0, 0, 0, 0, 6, 0, 0, 0, 4], #13
#                         [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0], #14
#                         [0, 0, 0, 0, 0, 0, 2, 0, 0, 12, 0, 8, 19, 0, 42, 0, 0, 0, 1, 0, 5, 0, 0, 0, 9], #15
#                         [0, 0, 8, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 0, 79, 0, 0, 0, 4, 0, 0, 2, 0, 0], #16
#                         [0, 0, 0, 0, 0, 1, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0], #17
#                         [0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 4, 69, 0, 0, 0, 0, 0, 0, 0], #18
#                         [0, 0, 0, 0, 2, 6, 0, 0, 0, 0, 7, 4, 0, 0, 0, 0, 0, 0, 80, 0, 0, 0, 0, 0, 0], #19
#                         [0, 0, 0, 0, 8, 0, 9, 0, 0, 8, 3, 0, 0, 0, 0, 0, 0, 0, 2, 68, 0, 0, 0, 0, 0], #20
#                         [0, 0, 2, 0, 0, 0, 0, 0, 0, 5, 0, 12, 4, 0, 1, 2, 0, 0, 0, 0, 69, 0, 0, 0, 4], #21
#                         [0, 0, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0], #22
#                         [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 0, 0], #23
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 92, 0], #24
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 6, 2, 0, 9, 0, 0, 0, 0, 0, 12, 0, 0, 0, 62]]) #25
#
# np.save('./res/M4.npy', conf_matrix)
#
# font_prop = FontProperties()
# font_prop.set_weight('bold')
# font_prop.set_size(20)
# font_prop.set_family('Times New Roman')
# fig, ax = plt.subplots(figsize=(8, 8))
# cax = ax.matshow(conf_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=conf_matrix.max())
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
#         plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color=text_color, fontsize=15)
#
# x = np.linspace(0, 24, 25)
# plt.tick_params(axis='x', top=False, labelbottom=True, labeltop=False)
# plt.xticks(x, fontsize=16)
# plt.yticks(x, fontsize=16)
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.xlabel('Predicted Label', fontproperties=font_prop)
# plt.ylabel('True Label', fontproperties=font_prop)
# plt.show()

def moving_average_filter(data, window_size):
    """
    滑动平均滤波函数
    :param data: 一维数组，表示要进行平滑处理的信号数据
    :param window_size: 整型，表示滑动窗口的大小
    :return: 一维数组，表示处理后的信号数据
    """
    # 构造滑动窗口
    window = np.ones(int(window_size)) / float(window_size)
    # 进行滑动平均滤波
    smoothed_data = np.convolve(data, window, mode='valid')
    return smoothed_data


font_properties = FontProperties()
font_prop = FontProperties()
font_prop.set_weight('bold')
font_prop.set_size(21)

f1 = np.load('./res/srff5.npy')
f2 = np.load('./res/srff6.npy')
f3 = np.load('./res/srff7.npy')
f4 = np.load('./res/srff8.npy')
plt.figure(figsize=(10, 6))
plt.plot(moving_average_filter(f1, 16), label="Device1")
plt.plot(moving_average_filter(f2, 16), label="Device2")
plt.plot(moving_average_filter(f3, 16), label="Device3")
plt.plot(moving_average_filter(f4, 16), label="Device4")
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.rcParams['font.sans-serif'] = 'Times New Roman'
#legend = plt.legend(loc='upper right', prop=font_prop)
plt.show()
