from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from dataloader.my_transforms import Rescale, RandomCrop, ToTensor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class VideoDataset(Dataset):
    """Video dataset."""

    def __init__(self, df, root_dir, transform=None, num_frames=4, gap_frame=2, split='train'):
        """
        Args:
            df (dataframe): my df
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df[df['split'] == split]
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.gap_frame = gap_frame
        self.split = split

    def __len__(self):
        return len(self.df)

    def __read_video(self, name_video):
        video_path = os.path.join(self.root_dir, self.split, 'videos', name_video)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        minutes = 0
        seconds = round(num_frames / fps)
        frame_id = int(fps*(minutes*60 + seconds))
        frames = []
        for i in range(0, frame_id, self.gap_frame):
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame is not None: 
                    frames.append(frame)
                if len(frames)==self.num_frames:
                    break
        assert len(frames) == self.num_frames, "do not have enough frames to dataloader"
        return np.array(frames[:self.num_frames])

    def __getitem__(self, idx):
        video_name = self.df.iloc[idx].fname
        frames = self.__read_video(video_name)
    
        sample = {'frames': frames, 'label': self.df.iloc[idx].liveness_score}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    video_dataset = VideoDataset(csv_file='../data/train/label.csv', \
                                root_dir='../data/train/videos', \
                                transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]), \
                                num_frames=4)
    for i in range(len(video_dataset)):
        sample = video_dataset[i] 
        print(sample['frames'].shape, sample['label'])
