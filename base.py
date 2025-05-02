import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from collections import deque
import os
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import csv
import time

from phone_predict import PhoneDetectorModel, phone_detector
from eye import eye_features
from hand_1 import hand_features
from head_pose import head_features




class MasterFeatureExtractor():
    def __init__( self, face_mesh_object, pose_object, frame_width=640, frame_height=480,
                    model_path=None, phone_window_size=30, mar_threshold=0.2, perclos_threshold=0.2,
                    baseline_yaw=5, baseline_pitch=-110, baseline_roll=10, yaw_threshold=15,
                    pitch_threshold=20, roll_threshold=20, min_away_duration=10, buffer_size=30,
                    event_window=30, proximity_threshold=0.2, alpha=1.8, beta=1.5,
                    default_wheel_box=[0.25, 0.92, 0.4, 0.6], window_size=30):
        
        self.face_mesh_object = face_mesh_object
        self.pose_object = pose_object
        self.phone_window_size = phone_window_size
        self.eye_features = eye_features( 
            frame_width=frame_width,
            frame_height=frame_height,
            mar_threshold=mar_threshold,
            perclos_threshold=perclos_threshold
        )
        self.head_features = head_features(
            frame_width=frame_width,
            frame_height=frame_height,
            baseline_yaw=baseline_yaw,
            baseline_pitch=baseline_pitch,
            baseline_roll=baseline_roll,
            yaw_threshold=yaw_threshold,
            pitch_threshold=pitch_threshold,
            roll_threshold=roll_threshold,
            min_away_duration=min_away_duration,
            buffer_size=buffer_size,
            event_window=event_window
        )
        self.hand_features = hand_features(
            frame_width=frame_width,
            frame_height=frame_height,
            proximity_threshold=proximity_threshold,
            alpha=alpha,
            beta=beta,
            default_wheel_box=default_wheel_box,
            window_size=window_size
        )
        self.phone_features = phone_detector(model_path, phone_window_size=phone_window_size)
        self.df = None

    def process_frame(self, frame, frame_id):
        # Process the frame with face mesh, hands, and pose models
        face_results = self.face_mesh_object.process(frame)
        pose_results = self.pose_object.process(frame)

        # Extract features
        eye_data = self.eye_features.calculate_eye_features(face_results,  frame_id)
        head_data = self.head_features.calculate_head_features(face_results, frame, frame_id)
        hand_data = self.hand_features.calculate_hand_features(pose_results, face_results)
        phone_data = self.phone_features.calculate_phone_feature(frame, frame_id)
        

        # Combine all features into a single dictionary
        combined_data = {
            "frame_id": frame_id,
            **eye_data,
            **head_data,
            **hand_data,
            **phone_data
        }
        # Append the combined data to the DataFrame
        if self.df is not None:
            self.df = pd.concat([self.df, pd.DataFrame([combined_data])], ignore_index=True)
        else:
            self.df = pd.DataFrame([combined_data])
        return combined_data
    
    @staticmethod
    def save_to_csv( filename, data):
        # Check if file exists
        file_exists = os.path.isfile(filename)
        
        # Open file in append mode if it exists, or create it if it doesn't
        with open(filename, mode='a' if file_exists else 'w', newline='') as file:
            # Get fieldnames from data dictionary
            fieldnames = list(data.keys())
            
            # Create CSV writer
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write header only if the file is being created
            if not file_exists:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(data)
    

if __name__ == "__main__":
    vid_pwd_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/"
    csv_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/video_labels.csv"
    start_time = time.time()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    model_path = "/home/harsh/Downloads/sem2/edgeai/edge ai project/phone_detector_model_final.pt"

    vid_data = pd.read_csv(csv_dir)
    mega_df = []
    for i in range(len(vid_data["video_address"])):
        print(f"Processing video {i+1}/{len(vid_data)}...")
        vid_file_name = vid_data["video_address"][i]
        vid_file_path = os.path.join(vid_pwd_dir, vid_file_name)
        cap = cv2.VideoCapture(vid_file_path)
        if not cap.isOpened():
            print(f"Error opening video file: {vid_file_path}")
            continue
        frame_id = 0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extractor = MasterFeatureExtractor(face_mesh, pose, frame_width=frame_width, frame_height=frame_height, model_path=model_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data = extractor.process_frame(frame_rgb, frame_id)
            frame_id += 1
            if frame_id >= 150:
                data["video_file_name"] = vid_file_name
                data["label"] = vid_data["label"][i]
                mega_df.append(data)
                #extractor.save_to_csv("output.csv", data)
        cap.release()
        cv2.destroyAllWindows()
        print(f"len(mega_df) = {len(mega_df)}")
    mega_df = pd.DataFrame(mega_df)
    mega_df.to_csv("output_final_2.csv", index=False)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
    print("Processing complete.")
