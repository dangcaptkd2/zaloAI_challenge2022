import cv2 
import os
import glob
from tqdm import tqdm

def save_video_2frames(path, save_folder):
    for name_video in tqdm(os.listdir(path)):

        name_folder = name_video.split('.')[0]
        path_folder = os.path.join(save_folder, name_folder)
        if not os.path.isdir(path_folder):
            os.mkdir(path_folder)
            
        video_path = os.path.join(path, name_video)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        minutes = 0
        seconds = round(num_frames / fps)
        frame_id = int(fps*(minutes*60 + seconds))
        frames = []
        for i in range(0, frame_id):
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                if frame is not None: 
                    cv2.imwrite(path_folder+'/'+str(i)+'.jpg', frame)
save_video_2frames('./data/train/videos', './data/train_frames')