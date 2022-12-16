from vivit import ViViT
from dataloader.my_dataloader import VideoDataset
from dataloader.my_transforms import Rescale, RandomCrop, ToTensor
from dataloader.my_dataset import load_csv_file

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import pandas as pd 

from tqdm import tqdm 
import numpy as np

class model():
    
    root_dir = './data/'
    df = load_csv_file(root_dir)
    lr = 0.001 
    criterion = nn.CrossEntropyLoss()
    batch_size = 16
    num_epoch = 5
    img_size = 224 
    patch_size = 8
    num_class = 2
    gap_frame = 1
    num_frames = 12
    net = ViViT(image_size=img_size, patch_size=patch_size, num_classes=num_class, num_frames=num_frames).cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    rescale_size = 128
    crop_size = 112
    video_dataset_train = VideoDataset(df=df, 
                                root_dir=root_dir, 
                                transform=transforms.Compose([Rescale(rescale_size), RandomCrop(crop_size), ToTensor()]),
                                num_frames=num_frames,
                                split='train')
    video_dataset_val = VideoDataset(df=df, 
                                root_dir=root_dir, 
                                transform=transforms.Compose([Rescale(rescale_size), RandomCrop(crop_size), ToTensor()]),
                                num_frames=num_frames,
                                split='val')
    video_dataset_test = VideoDataset(df=df, 
                                root_dir=root_dir, 
                                transform=transforms.Compose([Rescale(rescale_size), RandomCrop(crop_size), ToTensor()]),
                                num_frames=num_frames,
                                split='test')
    dataloader_train = DataLoader(video_dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_val = DataLoader(video_dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)
    dataloader_test = DataLoader(video_dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)
    def __init__(self, ):
        print("Dô init!")
        self.net = self.net.float()
        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)
    def train(self):
        for epoch in range(self.num_epoch):
            running_loss = 0.0
            for data in tqdm(self.dataloader_train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['frames'].cuda(), data['label'].cuda()
                # print(inputs.shape)
                # break

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs.float())
                loss = self.criterion(outputs, labels)
                # print(">>>loss:", loss)
                # exit()
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                # if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'epoch: {epoch + 1}, loss: {running_loss / len(self.dataloader):.3f}')
                # running_loss = 0.0

    def eval(self):
        pass 
    def predict(self):
        pass 

if __name__ == "__main__":
    print("alo?")
    model = model()
    print("alo?")
    model.train()