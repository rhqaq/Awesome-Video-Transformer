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
from torch.utils.data import DataLoader, TensorDataset
import random
from tqdm import tqdm
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, root_dir, chunk_size=32, num_chunks=10, max_frames=None, fps_step=4):
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.fps_step = fps_step
        self.max_frames = max_frames or chunk_size * num_chunks
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.clips = []
        for people_id in range(1, 15):
            for sit in ['glove', 'noglove']:
                video_file_1, video_file_2, time_file = self.get_file_paths(people_id, sit)
                if not (os.path.exists(video_file_1) and os.path.exists(video_file_2) and os.path.exists(time_file)):
                    continue
                ann = genfromtxt(time_file, delimiter=',')
                for action in ann:
                    action_id = int(action[0]) - 1
                    start, end = int(action[1]), int(action[2])
                    self.clips.extend([
                        (video_file_1, start, end, action_id),
                        (video_file_2, start, end, action_id)
                    ])
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_file, start, end, label = self.clips[idx]
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        starting_frame, end_frame = int(start * fps), int(end * fps)

        frames_list = []
        for frame_num in range(starting_frame, end_frame, self.fps_step):
            if len(frames_list) >= self.max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Change the color space from BGR to RGB
            frame = self.transform(frame)  # [3, 224, 224]
            frames_list.append(frame)

        cap.release()

        frames_tensor = torch.stack(frames_list)  # [T, 3, 224, 224]

        # Cut into chunks
        num_frames = frames_tensor.size(0)
        num_frames = num_frames - num_frames % self.chunk_size
        frames_tensor = frames_tensor[:num_frames]  # [num_chunks*chunk_size, 3, 224, 224]
        chunks_tensor = frames_tensor.view(-1, self.chunk_size, 3, 224, 224)  # [num_chunks, chunk_size, 3, 224, 224]
        if chunks_tensor.size(0) > self.num_chunks:
            chunks_tensor = chunks_tensor[:self.num_chunks]

        return chunks_tensor, torch.tensor([label] * chunks_tensor.size(0), dtype=torch.long)

    def get_file_paths(self, people_id, sit):
        if people_id < 10:
            video_file_1 = os.path.join(self.root_dir, f'0{people_id}_{sit}_cam1_720.mp4')
            video_file_2 = os.path.join(self.root_dir, f'0{people_id}_{sit}_cam2_720.mp4')
            time_file = os.path.join(self.root_dir, f'0{people_id}_{sit}.csv')
        else:
            video_file_1 = os.path.join(self.root_dir, f'{people_id}_{sit}_cam1_720.mp4')
            video_file_2 = os.path.join(self.root_dir, f'{people_id}_{sit}_cam2_720.mp4')
            time_file = os.path.join(self.root_dir, f'{people_id}_{sit}.csv')
        return video_file_1, video_file_2, time_file

        
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

def changeVideo2Tensor(data_path,time_file_path):
    if not os.access(time_file_path, os.F_OK):
        print(time_file_path,'时间文件不存在')
        return [],[]
    ann = genfromtxt(time_file_path, delimiter=',')
    if not os.access(data_path, os.F_OK):
        print(data_path,'视频文件不存在')
        return [],[]
    cap = cv2.VideoCapture(data_path)
    # print(cap.isOpened())
    # print('cap',cap)
    result_frames_list = []
    action_ids_list = []
    # print(ann)
    for action in ann:
        # print(action)
        start = action[1]
        end = action[2]
        action_id = int(action[0])-1

        fps = cap.get(cv2.CAP_PROP_FPS) #FPS of the video
        # print('fps',fps)
        starting_frame=int(start*fps) # value of the starting frame
        end_frame = int(end * fps) # value of the end frame
        # print('starting_frame',starting_frame)
        # print('end_frame',end_frame)
        cap.set(1,starting_frame) # Where frame_no is the frame you want
        # print(starting_frame)
        # print(action_id)
        frames_list = []
        for frame_num in range(starting_frame, end_frame):
            if frame_num % 4 == 0:
                ret, frame = cap.read()
                # print('ret',ret)
                # print('frame',frame)
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frames_list.append(frame)
                else:
                    break
        
        result_frames = torch.as_tensor(np.stack(frames_list)) # [temporal_dim, height, width, channel]
        result_frames = result_frames.permute(3, 0, 1, 2)
        result_frames = result_frames.float()/255.
        
        num_frames = result_frames.size(1)
        num_frames = num_frames - num_frames % 32
        
        result_frames = result_frames[:, :num_frames, :, :]
        num_chunks = result_frames.size(1) // 32
        

        chunks = torch.chunk(result_frames, chunks=num_chunks, dim=1)
        if num_chunks >10:
            num_chunks = 10
        chunks = chunks[:num_chunks]  
        print('num_chunks',num_chunks)
        action_ids = [action_id] * num_chunks
        action_ids = torch.tensor(action_ids, dtype=torch.long)
        
        # chunks_tensor = torch.stack(chunks, dim=0)
        action_ids_list.append(action_ids)
        result_frames_list.append(chunks)


        
    cap.release()
    return result_frames_list, action_ids_list

set_seed(2023) 

# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:1"
config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location=device)
# model = nn.DataParallel(model)
model.to(device)


all_frames = []
all_action_ids = []
# [batch_size, channel, temporal_dim, height, width]
for people_id in tqdm(range(1,15)):
    # all_frames = []
    # all_action_ids = []
    # if people_id == 2 or people_id==5:
    #     continue
    for sit in ['glove','noglove']:
        if people_id < 10:
            data_path_cam1 = '../data/0'+str(people_id)+'_'+sit+'_cam1_720.mp4'
            data_path_cam2 = '../data/0'+str(people_id)+'_'+sit+'_cam2_720.mp4'
            time_file_path = '../data/0'+str(people_id)+'_'+sit+'.csv'
        else:
            data_path_cam1 = '../data/'+str(people_id)+'_'+sit+'_cam1_720.mp4'
            data_path_cam2 = '../data/'+str(people_id)+'_'+sit+'_cam2_720.mp4'
            time_file_path = '../data/'+str(people_id)+'_'+sit+'.csv'
        
        result_frames_list_cam1,action_ids_list_cam1 = changeVideo2Tensor(data_path_cam1,time_file_path)
        result_frames_list_cam2,action_ids_list_cam2 = changeVideo2Tensor(data_path_cam2,time_file_path)
        # result_frames_list = result_frames_list_cam1 + result_frames_list_cam2
        # action_ids_list = action_ids_list_cam1 + action_ids_list_cam2
        all_frames += result_frames_list_cam1 + result_frames_list_cam2
        all_action_ids += action_ids_list_cam1 + action_ids_list_cam2
        # if len(all_frames) != 0:
        #     all_frames_tensor = torch.cat(all_frames, dim=0).float()
        #     all_action_ids_tensor = torch.cat(all_action_ids, dim=0)

        #     torch.save(all_frames_tensor, 'frames_train_{}.pth'.format(people_id))
        #     torch.save(all_action_ids_tensor, 'actions_train_{}.pth'.format(people_id))
print("load train data done")
        
# 将列表转换为Tensor

# all_frames_tensor = torch.cat(all_frames, dim=0).float()/255.
all_action_ids_tensor = torch.cat(all_action_ids, dim=0)

# torch.save(all_frames_tensor, 'frames_train.pth')
# torch.save(all_action_ids_tensor, 'actions_train.pth')

# 读取数据
# all_frames_tensor = torch.load('frames_train.pth')/255.
# all_action_ids_tensor = torch.load('actions_train.pth')
print("uniquetrain",torch.unique(all_action_ids_tensor))
train_dataset = VideoDataset(all_frames,all_action_ids_tensor)
print("train_dataset_prepared")

# print("all_frames_tensor",all_frames_tensor.shape)
print("all_action_ids_tensor",all_action_ids_tensor.shape)       

all_frames = []
all_action_ids = []
# [batch_size, channel, temporal_dim, height, width]
for people_id in range(15,18):
    # all_frames = []
    # all_action_ids = []
    # if people_id == 2 or people_id==5:
    #     continue
    for sit in ['glove','noglove']:
        if people_id < 10:
            data_path_cam1 = '../data/0'+str(people_id)+'_'+sit+'_cam1_720.mp4'
            data_path_cam2 = '../data/0'+str(people_id)+'_'+sit+'_cam2_720.mp4'
            time_file_path = '../data/0'+str(people_id)+'_'+sit+'.csv'
        else:
            data_path_cam1 = '../data/'+str(people_id)+'_'+sit+'_cam1_720.mp4'
            data_path_cam2 = '../data/'+str(people_id)+'_'+sit+'_cam2_720.mp4'
            time_file_path = '../data/'+str(people_id)+'_'+sit+'.csv'
        result_frames_list_cam1,action_ids_list_cam1 = changeVideo2Tensor(data_path_cam1,time_file_path)
        result_frames_list_cam2,action_ids_list_cam2 = changeVideo2Tensor(data_path_cam2,time_file_path)

        all_frames += result_frames_list_cam1 + result_frames_list_cam2
        all_action_ids += action_ids_list_cam1 + action_ids_list_cam2


print("load test data done")


all_action_ids_tensor = torch.cat(all_action_ids, dim=0)


print("uniquetest",torch.unique(all_action_ids_tensor))
test_dataset = VideoDataset(all_frames,all_action_ids_tensor)

# print("testall_frames_tensor",all_frames_tensor.shape)
print("testall_action_ids_tensor",all_action_ids_tensor.shape)  

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

classify_model = Mlp(1024,9,9).to(device)
# classify_model = nn.DataParallel(classify_model)  # 包装模型以使用DataParallel
classify_model.to(device)
backbone = model.backbone

optimizer = torch.optim.AdamW(classify_model.parameters(), lr=3e-4)

losses = []
# 开始训练
for epoch in range(100):  # 设定训练轮数为100，可以根据需要进行调整
    classify_model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 传递输入数据到模型
        backbone.eval()
        with torch.no_grad():
            features = backbone(inputs)
            features = features.mean(dim=[2,3,4])
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

with torch.no_grad():  # 在评估阶段，不需要计算梯度
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 传递输入数据到模型
        backbone.eval()
        with torch.no_grad():
            features = backbone(inputs)
            features = features.mean(dim=[2,3,4])
        outputs = classify_model(features)

        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test set: %d %%' % (100 * correct / total))


