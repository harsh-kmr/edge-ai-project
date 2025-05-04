from eye import eye_features
from hand_2 import hand_features
from head_pose import head_features
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def get_pose_estimation_features(video_path, video_name, pwd_dir):
    """
    Extracts pose estimation features from a video file.
    Args:
        video_path (str): Path to the video file.
        video_name (str): Name of the video file.
    Returns:
        list: A list of dictionaries containing pose estimation features for each frame.
        
    Landmarks Extracted:
        - left_eye: [33, 133, 160, 158, 144, 153]
        - right_eye: [362, 263, 385, 387, 380, 373]
        - left_pupil: [469, 470, 471, 472]
        - right_pupil: [474, 475, 476, 477]
        - mouth: [61, 73, 11, 303, 308, 403, 16, 180]
        - head: [1, 199, 130, 359, 61, 291]
        - left_shoulder: 11
        - right_shoulder: 12
        - left_elbow: 13
        - right_elbow: 14
        - left_wrist: 15
        - right_wrist: 16
        - left_palm: 17 (left pinky), 19 (left index)
        - right_palm: 18 (right pinky), 20 (right index)
        - left_hip: 23
        - right_hip: 24

        """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    eye_feature = eye_features(frame_height=frame_height, frame_width=frame_width)
    hand_feature = hand_features(frame_height=frame_height, frame_width=frame_width)
    head_feature = head_features(frame_height=frame_height, frame_width=frame_width)

    default_left_eye = {f"left_eye_{i}": None for i in range(6)}
    default_right_eye = {f"right_eye_{i}": None for i in range(6)}
    default_left_pupil = {f"left_pupil_{i}": None for i in range(4)}
    default_right_pupil = {f"right_pupil_{i}": None for i in range(4)}
    default_mouth = {f"mouth_{i}": None for i in range(8)}
    default_head = {f"head_{i}": None for i in range(6)}
    default_hand = {
            'left_shoulder': None,
            'right_shoulder': None,
            'left_elbow': None,
            'right_elbow': None,
            'left_wrist': None,
            'right_wrist': None,
            'left_palm': None,
            'right_palm': None,
            'left_hip': None,
            'right_hip': None,
            'left_eye': None,
            'right_eye': None,
            'eyes_center': None,
            'nose': None
        }
        

    landmark_list = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False 
        face_results = face_mesh.process(image_rgb)
        pose_results = pose.process(image_rgb)
        image_rgb.flags.writeable = True 
        left_eye, right_eye, left_pupil, right_pupil, mouth = eye_feature.get_landmarks(face_results)

        left_eye_landmarks = {f"left_eye_{i}": left_eye[i] for i in range(len(left_eye))} if left_eye is not None else default_left_eye
        right_eye_landmarks = {f"right_eye_{i}": right_eye[i] for i in range(len(right_eye))} if right_eye is not None else default_right_eye
        left_pupil_landmarks = {f"left_pupil_{i}": left_pupil[i] for i in range(len(left_pupil))} if left_pupil is not None else default_left_pupil
        right_pupil_landmarks = {f"right_pupil_{i}": right_pupil[i] for i in range(len(right_pupil))} if right_pupil is not None else default_right_pupil
        mouth_landmarks = {f"mouth_{i}": mouth[i] for i in range(len(mouth))} if mouth is not None else default_mouth

        hand_landmarks_result = hand_feature.get_landmarks(pose_results, face_results=face_results)
        hand_landmarks = hand_landmarks_result if hand_landmarks_result is not None else default_hand

        head_landmarks = default_head 
        if face_results and face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            head = head_feature.get_landmarks(face_landmarks=face_landmarks)
            if head is not None:
                head_landmarks = {f"head_{i}": head[i] for i in range(len(head))}

        combined_data = {
            **left_eye_landmarks,
            **right_eye_landmarks,
            **left_pupil_landmarks,
            **right_pupil_landmarks,
            **mouth_landmarks,
            **hand_landmarks, 
            **head_landmarks
        }


        combined_data["frame"] = f"{video_name}_frame_{frame_id}"
        # save the frames
        output_dir = os.path.join(pwd_dir, "frames")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_frame_path = os.path.join(output_dir, f"{video_name}_frame_{frame_id}.jpg")
        cv2.imwrite(output_frame_path, frame)

        landmark_list.append(combined_data)
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    return landmark_list


if __name__ == "__main__":
    pwd_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/"
    csv_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/landmark_dataset.csv"

    videos = os.listdir(pwd_dir)
    mega_df = []
    for video in videos:
        if video.endswith(".mp4"):
            video_path = os.path.join(pwd_dir, video)
            print(f"Processing video {video}...")
            landmark_list = get_pose_estimation_features(video_path, video, pwd_dir)
            mega_df.extend(landmark_list)
            print(f"Finished processing video {video}...")
    # Save the landmarks to a CSV file
    df = pd.DataFrame(mega_df)
    df.to_csv(csv_dir, index=False)
    print(f"Finished processing all videos...")





