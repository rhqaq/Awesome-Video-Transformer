# import matplotlib.pyplot as plt
import cv2 
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import os
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
from torchvision import transforms
import glob
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 根据cm的类型决定fmt的值
    if cm.dtype.kind in 'iu':
        fmt = 'd'
    else:
        fmt = '.2f'
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./confusion_matrix.png')
    
class VideoFrameDataset(Dataset):
    def __init__(self, base_dir, transform=None, num_frames=32, is_train=True):
        self.base_dir = base_dir
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []
        self.is_train = is_train
        self._prepare_samples()

    def _prepare_samples(self):
        class_folders = glob.glob(os.path.join(self.base_dir, '*'))  # 获取所有类别的文件夹
        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)  # 从文件夹路径中提取类别名
            people_folders = glob.glob(os.path.join(class_folder, '*'))  # 获取类别文件夹内的所有视频文件夹
            print("class_folder",class_folder)
            print("people_folders",len(people_folders))
            if self.is_train:
                people_folders = people_folders[:-8]
            else:
                people_folders = people_folders[-8:]
            
            for people_folder in people_folders:          
                # video_folders = glob.glob(os.path.join(people_folder, '*'))  # 获取视频文件夹            
                # for video_folder in video_folders:
                frame_files = sorted(glob.glob(os.path.join(people_folder, '*.jpg')))  # 获取视频文件夹内的所有帧文件
    
                if len(frame_files) > 0:
                    num_chunks = len(frame_files) // self.num_frames
                    if num_chunks >10:
                        num_chunks = 10
                    start_file_index = 0
                    added_chunks = 0
                    while start_file_index + self.num_frames < len(frame_files) and added_chunks < num_chunks:
                        frame_files_chunk = frame_files[start_file_index: start_file_index + self.num_frames]
                        self.samples.append((frame_files_chunk, class_name))
                        # print("frame_files_chunk",frame_files_chunk)
                        start_file_index += self.num_frames
                        added_chunks += 1
                        

    def __getitem__(self, index):
        frame_files, class_name = self.samples[index]
        frames = [torch.from_numpy(cv2.cvtColor(cv2.imread(frame_file),cv2.COLOR_BGR2RGB)) for frame_file in frame_files]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        frames_tensor = torch.stack(frames).permute(3, 0, 1, 2).float()/255.  # 将处理后的帧堆叠为一个张量
        return frames_tensor, int(class_name)

    def __len__(self):
        return len(self.samples)

        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

set_seed(2023) 

# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:2"
# config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
# checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
# input_dim = 768

config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'
input_dim = 1024

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location=device)
# model = nn.DataParallel(model)
model.to(device)


data_dir = '../data/data_prepared'
train_dataset = VideoFrameDataset(data_dir, num_frames=32, is_train=True)
print("train_dataset_prepared")

# print("all_frames_tensor",all_frames_tensor.shape)
print("trainset",len(train_dataset))       

test_dataset = VideoFrameDataset(data_dir, num_frames=32, is_train=False)

# print("testall_frames_tensor",all_frames_tensor.shape)
print("testset",len(test_dataset))  

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

classify_model = Mlp(input_dim,9,9).to(device)
# classify_model = nn.DataParallel(classify_model)  # 包装模型以使用DataParallel
classify_model.to(device)
backbone = model.backbone

optimizer = torch.optim.AdamW(classify_model.parameters(), lr=3e-4)
num_epochs = 1000
losses = []
# 开始训练
precomputed_features = []
precomputed_labels = []

# 通过backbone计算所有训练样本的特征，并保存到列表中
backbone.eval()  # 设置模型为评估模式
with torch.no_grad():
    for i, data in tqdm(enumerate(train_loader)):
        inputs, labels = data
        inputs = inputs.to(device)
        features = inputs
        # features = backbone(inputs)
        features = features.mean(dim=[2,3,4])  # 这部分可能需要根据你的实际需求进行调整
        print("features",features.shape)
        precomputed_features.append(features.cpu().detach().numpy())
        precomputed_labels.append(labels.numpy())

batch_size = 64
precomputed_batches = []
precomputed_features = np.concatenate(precomputed_features)
precomputed_labels = np.concatenate(precomputed_labels)
for i in range(0, len(precomputed_features), batch_size):
    batch_features = precomputed_features[i:i+batch_size]
    batch_labels = precomputed_labels[i:i+batch_size]
    precomputed_batches.append((batch_features, batch_labels))

# 在训练过程中直接使用预计算的特征
for epoch in range(num_epochs):
    classify_model.train()
    running_loss = 0.0
    for i in range(len(precomputed_batches)):
        features, labels = precomputed_batches[i]
        features = torch.from_numpy(features).to(device)
        labels = torch.from_numpy(labels).to(device)

        outputs = classify_model(features)
         # 计算损失
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        # 反向传播并优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("loss",loss)
        running_loss += loss.item()
    average_loss = running_loss / len(train_loader)
    losses.append(average_loss)
    # 打印每个epoch的loss
    print('[%d] Loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

# for epoch in range(1):  # 设定训练轮数为100，可以根据需要进行调整
#     classify_model.train()  # 设置模型为训练模式
#     running_loss = 0.0
#     for i, data in enumerate(train_loader):
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         # 传递输入数据到模型
#         backbone.eval()
#         with torch.no_grad():
#             features = backbone(inputs)
#             features = features.mean(dim=[2,3,4])
#             # print(features.shape)
#         outputs = classify_model(features)

#         # 计算损失
#         loss_fn = nn.CrossEntropyLoss()
#         loss = loss_fn(outputs, labels)

#         # 反向传播并优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print("loss",loss)
#         running_loss += loss.item()
#     average_loss = running_loss / len(train_loader)
#     losses.append(average_loss)
#     # 打印每个epoch的loss
#     print('[%d] Loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

print('Finished Training')
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')


# 开始评估
classify_model.eval()  # 设置模型为评估模式
correct = 0
total = 0

torch.cuda.empty_cache()

# 初始化混淆矩阵
n_classes = 9  # 需要根据你的任务进行修改
cm = np.zeros((n_classes, n_classes))

torch.cuda.empty_cache()

with torch.no_grad():  # 在评估阶段，不需要计算梯度
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 传递输入数据到模型
        backbone.eval()
        with torch.no_grad():
            # features = backbone(inputs)
            features = inputs
            features = features.mean(dim=[2,3,4])
        outputs = classify_model(features)

        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新混淆矩阵
        cm += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=range(n_classes))

print('Accuracy on the test set: %d %%' % (100 * correct / total))

for i in range(n_classes):
    tp = cm[i, i]
    fp_fn = np.sum(cm[:, i])
    accuracy = tp / fp_fn if fp_fn > 0 else 0.
    print('Accuracy of class %d : %d %%' % (i, 100 * accuracy))
# 显示混淆矩阵
plot_confusion_matrix(cm, classes=range(n_classes))

