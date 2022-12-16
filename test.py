import pandas as pd 
import cv2 
import os 
from tqdm import tqdm

df = pd.read_csv('./data/train/label.csv')  

def read_video(path):
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        minutes = 0
        seconds = 5
        frame_id = int(fps*(minutes*60 + seconds))
        error=0
        frames = []
        for i in range(0, frame_id, 2):
            try:
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                a = frame.shape
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append(frame)
            except:
                print(path)
                error+=1
        if len(frames) < 16:
            print(">>>>>> Khong dat")

r = []
for index, row in tqdm(df.iterrows()):
    path_video = os.path.join('./data/train/videos', row['fname'])
    num_frame = read_video(path=path_video)
#     r.append(num_frame)
# print(min(r))

# a = read_video("./data/train/videos/31.mp4")

# class hello():
#     a = 10
#     b = a+10
# c = hello()
# print(c.a)
# print(c.b)