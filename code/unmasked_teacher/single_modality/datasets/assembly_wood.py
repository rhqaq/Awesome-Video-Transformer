import os
import io
import cv2
import numpy as np
import torch
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter, 
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
)
from .volume_transforms import ClipToTensor
import glob

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False


class WoodDataset(Dataset):
    def __init__(self, base_dir, transform=None, num_frames=32, mode='train'):
        self.base_dir = base_dir
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []
        self.mode = mode
        self._prepare_samples()

    def _prepare_samples(self):
        class_folders = glob.glob(os.path.join(self.base_dir, '*'))  # 获取所有类别的文件夹
        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)  # 从文件夹路径中提取类别名
            people_folders = glob.glob(os.path.join(class_folder, '*'))  # 获取类别文件夹内的所有视频文件夹
            print("class_folder",class_folder)
            print("people_folders",len(people_folders))
            if (self.mode == 'train'):
                people_folders = people_folders[:-12]
            elif (self.mode == 'validation'):
                people_folders = people_folders[-12:-8]
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
        if self.mode == 'train':
            return frames_tensor, int(class_name), index, {}
        elif self.mode=='validation':
            return frames_tensor, int(class_name), {}
        else:
            return frames_tensor, int(class_name), {}, {},{}

    def __len__(self):
        return len(self.samples)