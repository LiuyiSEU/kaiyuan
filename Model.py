import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        a = self.channel_attention(x)
        b = self.spatial_attention(x)
        return a,b


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        output1 = F.relu(self.bn1(self.conv1(x)))
        output1 = self.bn2(self.conv2(output1))
        if self.conv3:
            output2 = self.conv3(x)
            output = F.relu(output1 + output2)
            return output
        else:
            output = F.relu(output1)
        return output


def distillation_loss(outputs_teacher, outputs_student, labels, temperature, alpha):
    soft_teacher = F.softmax(outputs_teacher / temperature, dim=1)
    soft_student = F.log_softmax(outputs_student / temperature, dim=1)
    loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    classification_loss = F.cross_entropy(outputs_student, labels)
    total_loss = alpha * loss + (1 - alpha) * classification_loss
    #print('loss:')
    #print(loss)
    #print(classification_loss)
    return total_loss


def huber_loss(predictions, targets, delta=1.0):
    residual = torch.abs(predictions - targets)
    condition = residual < delta
    squared_loss = 0.5 * torch.pow(residual, 2)
    linear_loss = delta * (residual - 0.5 * delta)
    loss = torch.where(condition, squared_loss, linear_loss)
    return loss.mean()

def cosine_similarity_loss(student_features, teacher_features):
    
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)

    similarity = F.cosine_similarity(student_features, teacher_features, dim=1)
    
    loss = 1.0 - similarity.mean()
    
    return loss



class teacherNet(nn.Module):
    def __init__(self):
        super(teacherNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.res_block1 = nn.Sequential(res_block(16, 16, stride=1),
                                        res_block(16, 16, stride=1))

        self.res_block2 = nn.Sequential(res_block(16, 32, use_1x1conv=True, stride=2),
                                        res_block(32, 32, stride=1))

        self.res_block3 = nn.Sequential(res_block(32, 64, use_1x1conv=True, stride=2),
                                        res_block(64, 64, stride=1))

        self.res_block4 = nn.Sequential(res_block(64, 128, use_1x1conv=True, stride=2),
                                        res_block(128, 128, stride=1))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 25)
        self.att = SpatialAttention()

    def forward(self, x):
        in_size = x.size(0)
        output = self.conv1(x)
        output = self.res_block1(output)
        att1 = self.att(output)
        output = att1 * output
        output = self.res_block2(output)
        att2 = self.att(output)
        output = att2 * output
        output = self.res_block3(output)
        #att3 = self.att(output)
        #output = att3 * output
        output = self.res_block4(output)
        #att4 = self.att(output)
        #output = att4 * output
        output = self.avgpool(output)
        output = output.view(in_size, -1)
        f = output 
        output = self.fc(output)
        return output, f


class studentNet(nn.Module):
    def __init__(self):
        super(studentNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.res_block1 = nn.Sequential(res_block(16, 16, stride=1),
                                        nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=2, padding=3),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU())

        self.res_block2 = nn.Sequential(res_block(16, 32, use_1x1conv=True, stride=2),
                                        nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=2, padding=3),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU())

        self.res_block3 = nn.Sequential(res_block(32, 64, use_1x1conv=True, stride=2),
                                        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU())

        self.res_block4 = nn.Sequential(res_block(64, 128, use_1x1conv=True, stride=2),
                                        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=2, padding=3),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU())


        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 25)

        

    def forward(self, x):
        in_size = x.size(0)
        output = self.conv1(x)
        output = self.res_block1(output)
        output = self.res_block2(output)
        output = self.res_block3(output)
        output = self.res_block4(output)
        output = self.avgpool(output)
        output = output.view(in_size, -1)
        f = output
        output = self.fc(output)
        return output, f


model = studentNet().to('cuda')
summary(model, (2, 4096))