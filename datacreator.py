from preprocessing import Preprocessing
from base import MasterFeatureExtractor
import os
import cv2
import pandas as pd
import mediapipe as mp

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
csv_path = os.path.join(video_output_dir, 'video_labels.csv')
vid_data = pd.read_csv(csv_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp.solutions.hands.Hands()
mega_df = pd.DataFrame()
for i in range(len(vid_data["video_address"])):
    vid_file_name = vid_data["video_address"][i]
    vid_file_path = os.path.join(video_output_dir, vid_file_name)
    cap = cv2.VideoCapture(vid_file_path)
    if not cap.isOpened():
        print(f"Error opening video file: {vid_file_path}")
        continue
    frame_id = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing video {vid_file_name} with resolution {frame_width}x{frame_height}")
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    extractor = MasterFeatureExtractor(face_mesh, hands, frame_width=frame_width, frame_height=frame_height)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results, hand_results = extractor.process_frame(frame_rgb, frame_id)
        frame_id += 1
    df = extractor.df
    # drop first 150 rows to have consistent time features
    df = df.iloc[150:]
    df["label"] = vid_data["label"][i]
    df["video_file_name"] = vid_file_name
    mega_df = pd.concat([mega_df, df], ignore_index=True)
    cap.release()
    print(f"Processed video {vid_file_name} with {len(df)} frames.")
    cv2.destroyAllWindows()
mega_df.to_csv("total_data_3.csv", index=False)
