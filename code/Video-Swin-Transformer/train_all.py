import cv2 
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmaction.models import I3DHead
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
from torchvision import transforms
import glob
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    plt.savefig('./noglove_cam1_confusion_matrix.png')
    
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


class SplitBackbone(nn.Module):
    def __init__(self, backbone):
        super(SplitBackbone, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

set_seed(2023) 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cuda:0"
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
input_dim = 768

# config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'
# input_dim = 1024

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint)
backbone_ = model.backbone
# model = nn.DataParallel(model)
# model.to(device)

num_frames = 16

data_dir = '../data/data_prepared'
train_dataset = VideoFrameDataset(data_dir, num_frames=num_frames, is_train=True)
print("train_dataset_prepared")

# print("all_frames_tensor",all_frames_tensor.shape)
print("trainset",len(train_dataset))       

test_dataset = VideoFrameDataset(data_dir, num_frames=num_frames, is_train=False)

# print("testall_frames_tensor",all_frames_tensor.shape)
print("testset",len(test_dataset))  

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

classify_model = I3DHead(in_channels=768,
        num_classes=9,
        spatial_type='avg',
        dropout_ratio=0.5)
classify_model = nn.DataParallel(classify_model)  # 包装模型以使用DataParallel
classify_model.to(device)

# model.to(device)

backbone = SplitBackbone(backbone_)
backbone = nn.DataParallel(backbone)
backbone.to(device)
accumulation_steps = 16  # 您可以根据自己的需要设定这个值
optimizer_bone = torch.optim.AdamW(backbone.parameters(), lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05)
optimizer = torch.optim.AdamW(classify_model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05)
num_epochs = 30
warmup_epochs = 2.5  # Define warmup epochs
total_steps = int(len(train_loader)/accumulation_steps * num_epochs)  # Assuming dataloader is defined
warmup_steps = int(len(train_loader)/accumulation_steps * warmup_epochs)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0)
scheduler_bone = CosineAnnealingLR(optimizer_bone, T_max=total_steps - warmup_steps, eta_min=0)

losses = []
# 开始训练
precomputed_features = []
precomputed_labels = []

# 设定累积的步数


# 训练
for epoch in tqdm(range(num_epochs)):  
    classify_model.train() 
    backbone.train()
    running_loss = 0.0
    
    optimizer.zero_grad()
    optimizer_bone.zero_grad()
    
    for i, data in tqdm(enumerate(train_loader)):
        global_step = epoch * len(train_loader) + i
        if global_step < warmup_steps:
            warmup_factor = global_step / float(warmup_steps)
            lr = 3e-5 * warmup_factor
            for param_group in optimizer_bone.param_groups:
                param_group['lr'] = lr
            lr = 3e-4 * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
            scheduler_bone.step()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        backbone.eval()
        features = backbone(inputs)
        # features = features.mean(dim=[2,3,4])
        outputs = classify_model(features)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        loss.backward()  # 计算梯度，但不立即更新模型权重
        
        if (i+1) % accumulation_steps == 0:  # 每 accumulation_steps 步，我们才更新模型权重
            optimizer_bone.step()
            optimizer.step()
            optimizer.zero_grad()
            optimizer_bone.zero_grad()

        running_loss += loss.item() * inputs.size(0)  # 注意这里，我们需要累积的是“总损失”，所以要乘以 batch size

    average_loss = running_loss / len(train_dataset)  # 注意这里，我们需要的是平均损失，所以要除以数据集的总长度，而不是数据加载器的长度
    losses.append(average_loss)

    print('[%d] Loss: %.3f' % (epoch + 1, average_loss))

    torch.save(classify_model.state_dict(), './checkpoints/noglove_cam1_newclassify_model_epoch_{}.pth'.format(epoch))
    torch.save(backbone.state_dict(), './checkpoints/noglove_cam1_newbackbone_epoch_{}.pth'.format(epoch))   



    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss_noglove_cam1.png')

print('Finished Training')
# 开始评估
classify_model.eval()  # 设置模型为评估模式
backbone.eval()
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
        
        
        features = backbone(inputs)
        # features = features.mean(dim=[2,3,4])
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

