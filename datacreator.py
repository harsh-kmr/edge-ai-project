from preprocessing import Preprocessing
#from base import MasterFeatureExtractor
import os
import cv2
import pandas as pd

video_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/Raw data"
video_output_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data"

#Frame width: 1920
#Frame height: 1080
processor = Preprocessing(
    image_size=( 960, 540),
    video_dir=video_dir,
    video_output_dir=video_output_dir,
    video_fps=30,
    video_file_type=".mp4",
    video_duration=30,
    save_video_flag=True,
    convert_color_flag=True,
    color_mode='rgb',
    debug_flag=False
)

processor()

videos = os.listdir(video_output_dir)
# Create a dataframe with video file name and label
data = []
for video in videos:
    if video.endswith('.mp4'):
        if 'blink' in video.lower():
            label = 'blink'
        elif 'yawn' in video.lower():
            label = 'yawn'
        elif 'normal' in video.lower():
            label = 'normal'
        elif 'distracted' in video.lower():
            label = 'distracted'
        elif 'phone' in video.lower():
            label = 'phone'
        elif 'sleep' in video.lower():
            label = 'sleep'
        else:
            label = 'unknown'
        
        data.append({'video_address': video, 'label': label})

df = pd.DataFrame(data)
print(f"Created dataframe with {len(df)} videos")
print(df['label'].value_counts())

# Save dataframe to csv
df.to_csv(os.path.join(video_output_dir, 'video_labels.csv'), index=False)

# we will use this dataframe to process the videos
# and extract features
# and collect the data in a new dataframe