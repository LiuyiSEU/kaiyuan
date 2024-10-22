from scipy.io import loadmat
import predict
import trainTeacher, trainStudent, classification, trainS
import torch
from PIL import Image
from torchvision import transforms
import scipy.io as sio

#trainTeacher.train()
trainStudent.train()
#classification.train()


model_path = '/home/liuyi/RFF/res/teacher_model.pth'
data_path = '/home/liuyi/ORI_max/val'
#predict.model_predict(model_path, data_path, 25)

rec_path = '/home/liuyi/RFF/RFFdata/ORI_1d/val'
#trainStudent.extract(rec_path)
