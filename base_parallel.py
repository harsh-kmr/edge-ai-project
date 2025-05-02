import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from collections import deque
import os
from PIL import Image
import csv
import time
import multiprocessing as mp_proc
from functools import partial

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



import os
import time
import cv2
import pandas as pd
import mediapipe as mp
import multiprocessing as mp_proc
from functools import partial
import numpy as np
import uuid

# Assuming MasterFeatureExtractor is your custom class
# If it's imported from somewhere, you'd need to add that import
class MasterFeatureExtractor:
    def __init__(self, face_mesh, pose, frame_width, frame_height, model_path):
        self.face_mesh = face_mesh
        self.pose = pose
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model_path = model_path
        # Add any other initialization needed for your extractor

    def process_frame(self, frame_rgb, frame_id):
        # Placeholder for your actual processing logic
        # Return the data and landmarks as in your original code
        data = {"frame_id": frame_id}
        left_eye_landmarks = None
        right_eye_landmarks = None
        left_pupil_landmarks = None
        right_pupil_landmarks = None
        mouth_landmarks = None
        hand_landmarks = None
        head_landmarks = None
        
        # Your actual processing logic here
        
        return data, left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks, head_landmarks


def process_video(video_index, vid_pwd_dir, vid_data):
    """Process a single video and return extracted features"""
    # Generate a unique identifier for this worker
    worker_id = f"worker-{video_index}-{uuid.uuid4().hex[:8]}"
    
    # Get video information
    vid_file_name = vid_data["video_address"].iloc[video_index]
    vid_label = vid_data["label"].iloc[video_index]
    vid_file_path = os.path.join(vid_pwd_dir, vid_file_name)
    
    print(f"[{worker_id}] Starting processing of video {video_index+1}: {vid_file_name}")
    
    # Initialize MediaPipe for this process
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Initialize model (use model_path from the data passed)
    model_path = vid_data["model_path"].iloc[0]  # Assuming model_path is the same for all videos
    
    # Process video
    video_results = []
    start_time = time.time()
    
    try:
        cap = cv2.VideoCapture(vid_file_path)
        if not cap.isOpened():
            print(f"[{worker_id}] Error opening video file: {vid_file_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[{worker_id}] Video info: {total_frames} frames, {frame_width}x{frame_height}, {fps} FPS")
        
        extractor = MasterFeatureExtractor(face_mesh, pose, frame_width=frame_width, 
                                          frame_height=frame_height, model_path=model_path)
        
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Here we call the actual feature extraction with all needed parameters
            # This should match your original implementation exactly
            data, left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks, head_landmarks = extractor.process_frame(frame_rgb, frame_id)
            
            # Add the video metadata to the extracted features
            data["video_file_name"] = vid_file_name
            data["label"] = vid_label
            
            # Add to results
            video_results.append(data)
            
            frame_id += 1
            
            # Report progress periodically
            if frame_id % 1000 == 0:
                elapsed = time.time() - start_time
                frames_per_sec = frame_id / elapsed if elapsed > 0 else 0
                print(f"[{worker_id}] Processed {frame_id}/{total_frames} frames ({frames_per_sec:.2f} fps)")
                
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"[{worker_id}] Completed video {video_index+1}: {vid_file_name}")
        print(f"[{worker_id}] Processed {frame_id} frames in {elapsed:.2f} seconds ({frame_id/elapsed:.2f} fps)")
        print(f"[{worker_id}] Collected {len(video_results)} data points")
        
    except Exception as e:
        print(f"[{worker_id}] Error processing video {vid_file_name}: {str(e)}")
    
    # Release resources
    face_mesh.close()
    pose.close()
    
    return video_results


def save_chunk_to_csv(chunk_df, output_path, chunk_id):
    """Save a chunk of data to a temporary CSV file"""
    temp_file = f"{output_path}.part{chunk_id}.csv"
    chunk_df.to_csv(temp_file, index=False)
    return temp_file


def merge_csv_files(file_list, output_path):
    """Merge multiple CSV files into one"""
    # Read and combine all dataframes
    print(f"Merging {len(file_list)} CSV files...")
    
    # Use pandas to read and concatenate all files
    all_dfs = []
    for file in file_list:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
            # Delete temporary file after reading
            os.remove(file)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    if not all_dfs:
        print("No valid data files to merge!")
        return None
        
    # Combine all dataframes
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to final output file
    final_df.to_csv(output_path, index=False)
    print(f"Combined data saved to {output_path} with {len(final_df)} rows")
    return final_df


if __name__ == "__main__":
    vid_pwd_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/"
    csv_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/video_labels.csv"
    model_path = "/home/harsh/Downloads/sem2/edgeai/edge ai project/phone_detector_model_final.pt"
    output_file = "output_parallel.csv"
    
    start_time = time.time()
    
    # Read video data
    vid_data = pd.read_csv(csv_dir)
    num_videos = len(vid_data)
    
    # Add model path to vid_data to pass it to workers
    vid_data["model_path"] = model_path
    
    # Determine number of processes to use
    num_processes = min(mp_proc.cpu_count(), num_videos)
    print(f"Using {num_processes} parallel processes to process {num_videos} videos")
    
    # Create a smaller dataframe with just the needed columns for multiprocessing
    worker_data = vid_data[["video_address", "label", "model_path"]].copy()
    
    # Process videos in parallel
    process_video_partial = partial(process_video, 
                                    vid_pwd_dir=vid_pwd_dir,
                                    vid_data=worker_data)
    
    # Use multiprocessing to process videos
    with mp_proc.Pool(processes=num_processes) as pool:
        results = pool.map(process_video_partial, range(num_videos))
    
    # Merge all results into one data structure
    print("Processing complete. Merging results...")
    
    # Create a list to keep track of temporary files
    temp_files = []
    
    # Process results by chunks to avoid memory issues with very large datasets
    chunk_id = 0
    chunk_size = 100000  # Adjust based on available memory
    
    for video_results in results:
        if not video_results:
            continue
            
        # Convert to DataFrame and save as temporary file
        chunk_df = pd.DataFrame(video_results)
        temp_file = save_chunk_to_csv(chunk_df, output_file, chunk_id)
        temp_files.append(temp_file)
        chunk_id += 1
    
    # Merge all temporary CSV files
    final_df = merge_csv_files(temp_files, output_file)
    
    # Done
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    if final_df is not None:
        print(f"Total frames processed: {len(final_df)}")
    print("Processing complete.")