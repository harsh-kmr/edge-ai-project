import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from collections import deque
import os
from PIL import Image
import csv
import time

#from phone_predict import PhoneDetectorModel, phone_detector
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
        self.width = frame_width
        self.height = frame_height
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
        #self.phone_features = phone_detector(model_path, phone_window_size=phone_window_size)
        self.df = None

    def process_frame(self, frame, frame_id):
        # Process the frame with face mesh, hands, and pose models
        face_results = self.face_mesh_object.process(frame)
        pose_results = self.pose_object.process(frame)

        # Extract features
        eye_data = self.eye_features.calculate_eye_features(face_results,  frame_id)
        head_data = self.head_features.calculate_head_features(face_results, frame, frame_id)
        hand_data = self.hand_features.calculate_hand_features(pose_results, face_results)
        #phone_data = self.phone_features.calculate_phone_feature(frame, frame_id)
        

        # Combine all features into a single dictionary
        combined_data = {
            #"frame_id": frame_id,
            **eye_data,
            **head_data,
            **hand_data,
            #**phone_data
        }
        left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks , mouth_landmarks= self.eye_features.get_landmarks(face_results)
        hand_landmarks = self.hand_features.extract_hand_position(pose_results)
        if hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks:
            face_landmarks_ = face_results.multi_face_landmarks[0]
            head_landmarks = self.head_features.get_landmarks(face_landmarks_)
        else:
            head_landmarks = None
        return combined_data, left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks, head_landmarks

    
    def get_feature_for_model(self, frame, frame_id):
        combined_data, left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks, head_landmarks = self.process_frame(frame, frame_id)
        # Ordered list of keys
        expected_keys = [
            "left_eye_aspect_ratio",
            "right_eye_aspect_ratio",
            "left_eye_aspect_ratio_3d",
            "right_eye_aspect_ratio_3d",
            "left_eye_pupil_distance",
            "right_eye_pupil_distance",
            "no_visible_eyes",
            "left_eye_closed",
            "right_eye_closed",
            "mouth_aspect_ratio",
            "eye_closure_during_yawn",
            "left_eye_pupil_movement",
            "left_eye_pupil_variance",
            "right_eye_pupil_movement",
            "right_eye_pupil_variance",
            "num_blinks",
            "ear_mean",
            "ear_variance",
            "perclos",
            "yaw",
            "pitch",
            "roll",
            "yaw_variance",
            "pitch_variance",
            "roll_variance",
            "head_away_duration",
            "head_away_event_count",
            "distraction_confidence",
            "yaw_deviation",
            "pitch_deviation",
            "roll_deviation",
            "left_wrist_x",
            "left_wrist_y",
            "left_wrist_z",
            "left_palm_x",
            "left_palm_y",
            "left_palm_z",
            "right_wrist_x",
            "right_wrist_y",
            "right_wrist_z",
            "right_palm_x",
            "right_palm_y",
            "right_palm_z",
            "left_hand_eye_depth",
            "right_hand_eye_depth",
            "left_elbow_angle",
            "right_elbow_angle",
            "left_hand_off_wheel",
            "left_hand_off_wheel_duration",
            "right_hand_off_wheel",
            "right_hand_off_wheel_duration",
            "left_hand_distance",
            "right_hand_distance",
            "left_hand_distance_var",
            "right_hand_distance_var"
        ]
        feature_list = [0 if combined_data.get(key, None) is None else combined_data.get(key, None) for key in expected_keys]

        
        
        return feature_list, left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks, head_landmarks

    
    

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
            data , left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks, head_landmarks = extractor.process_frame(frame_rgb, frame_id)
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
