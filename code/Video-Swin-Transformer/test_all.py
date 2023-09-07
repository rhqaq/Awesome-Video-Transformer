import cv2 
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmaction.models import I3DHead
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
from collections import OrderedDict

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
    plt.savefig('./confusion_matrix_new_16.png')
    
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



def load_model(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Create a new ordered dictionary to store the modified key-value pairs
    new_state_dict = OrderedDict()

    # Iterate over all items in the checkpoint
    for k, v in checkpoint.items():
        # If the name of the key starts with "module.", remove this prefix
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.' prefix
        else:
            name = k
        # Add the modified key and the original value to the new dictionary
        new_state_dict[name] = v

    # Use the new state_dict to load the model
    model.load_state_dict(new_state_dict)

    return model



set_seed(2023) 

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
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


data_dir = '../data/data_prepared'
train_dataset = VideoFrameDataset(data_dir, num_frames=16, is_train=True)
print("train_dataset_prepared")

# print("all_frames_tensor",all_frames_tensor.shape)
print("trainset",len(train_dataset))       

test_dataset = VideoFrameDataset(data_dir, num_frames=16, is_train=False)

# print("testall_frames_tensor",all_frames_tensor.shape)
print("testset",len(test_dataset))  

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

classify_model = I3DHead(in_channels=768,
        num_classes=9,
        spatial_type='avg',
        dropout_ratio=0.5)
classify_model = load_model(classify_model, './checkpoints/noglove_cam1_newclassify_model_epoch_19.pth')
# classify_model.load_state_dict(torch.load('./checkpoints/classify_model_epoch_10.pth'))
classify_model = nn.DataParallel(classify_model)  # 包装模型以使用DataParallel
classify_model.to(device)

# model.to(device)

backbone = SplitBackbone(backbone_)
backbone = load_model(backbone, './checkpoints/noglove_cam1_newbackbone_epoch_19.pth')
# backbone.load_state_dict(torch.load('./checkpoints/backbone_epoch_10.pth'))
backbone = nn.DataParallel(backbone)
backbone.to(device)

optimizer_bone = torch.optim.AdamW(backbone.parameters(), lr=3e-5)
optimizer = torch.optim.AdamW(classify_model.parameters(), lr=3e-4)
num_epochs = 30
losses = []
# 开始训练
precomputed_features = []
precomputed_labels = []

# 设定累积的步数
accumulation_steps = 32  # 您可以根据自己的需要设定这个值


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

