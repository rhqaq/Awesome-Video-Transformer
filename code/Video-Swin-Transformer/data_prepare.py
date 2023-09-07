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

def save_frames_as_images(video_file, output_dir, time_file_path, video_id):
    if not os.access(time_file_path, os.F_OK):
        print(time_file_path,'时间文件不存在')
        return [],[]
    ann = genfromtxt(time_file_path, delimiter=',')
    if not os.access(video_file, os.F_OK):
        print(video_file,'视频文件不存在')
        return [],[]
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: ", fps)
    for action in ann:
        # print(action)
        start = action[1]
        end = action[2]
        action_id = int(action[0])-1

        starting_frame=int(start*fps) # value of the starting frame
        end_frame = int(end * fps) # value of the end frame
        
        cap.set(1,starting_frame)
        
        
        frame_num = 0
        # 创建路径，路径名称为 action_id/video_id
        _output_dir = os.path.join(output_dir, str(action_id), str(video_id))
        # print(_output_dir)
        os.makedirs(_output_dir, exist_ok=True)  # 确保目录存在
        for k in range(starting_frame, end_frame):
            if k % 16 == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                # 每个帧的文件名为 frame_num.jpg
                frame_file = os.path.join(_output_dir, f"{frame_num}.jpg")
                # print(frame_file)
                cv2.imwrite(frame_file, frame)
                frame_num += 1
    cap.release()


# all data        
# for people_id in tqdm(range(1, 18)):
#     for sit in ['glove', 'noglove']:
#         if people_id < 10:
#             data_path_cam1 = '../data/0'+str(people_id)+'_'+sit+'_cam1_720.mp4'
#             data_path_cam2 = '../data/0'+str(people_id)+'_'+sit+'_cam2_720.mp4'
#             time_file_path = '../data/0'+str(people_id)+'_'+sit+'.csv'
#         else:
#             data_path_cam1 = '../data/'+str(people_id)+'_'+sit+'_cam1_720.mp4'
#             data_path_cam2 = '../data/'+str(people_id)+'_'+sit+'_cam2_720.mp4'
#             time_file_path = '../data/'+str(people_id)+'_'+sit+'.csv'

#         save_frames_as_images(data_path_cam1, '../data', time_file_path, f"people_{people_id}_{sit}_cam1")
#         save_frames_as_images(data_path_cam2, '../data', time_file_path, f"people_{people_id}_{sit}_cam2")



for people_id in tqdm(range(1, 18)):
    for sit in ['glove']:
        if people_id < 10:
            data_path_cam1 = '../data/0'+str(people_id)+'_'+sit+'_cam1_720.mp4'
            data_path_cam2 = '../data/0'+str(people_id)+'_'+sit+'_cam2_720.mp4'
            time_file_path = '../data/0'+str(people_id)+'_'+sit+'.csv'
        else:
            data_path_cam1 = '../data/'+str(people_id)+'_'+sit+'_cam1_720.mp4'
            data_path_cam2 = '../data/'+str(people_id)+'_'+sit+'_cam2_720.mp4'
            time_file_path = '../data/'+str(people_id)+'_'+sit+'.csv'

        save_frames_as_images(data_path_cam1, '../data/data_prepared_glove_cam1', time_file_path, f"people_{people_id}_{sit}_cam1")
        # save_frames_as_images(data_path_cam2, '../data/data_prepared_glove_cam1', time_file_path, f"people_{people_id}_{sit}_cam2")
