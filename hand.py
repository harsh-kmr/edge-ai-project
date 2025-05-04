import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from collections import deque

class hand_features():
    def __init__(self, frame_width=640, frame_height=480, 
                 proximity_threshold=0.2, 
                 alpha=1.8, beta=1.5,  
                 default_wheel_box=[0.25, 0.92, 0.4, 0.6],  
                 window_size=150): 
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.proximity_threshold = proximity_threshold
        self.default_wheel_box = default_wheel_box 
        self.alpha = alpha  
        self.beta = beta    

        self.left_hand_near_face_history = deque(maxlen=window_size)
        self.right_hand_near_face_history = deque(maxlen=window_size)
        self.left_hand_off_wheel_history = deque(maxlen=window_size)
        self.right_hand_off_wheel_history = deque(maxlen=window_size)
        
        self.current_wheel_box = default_wheel_box
    
    def calculate_wheel_box(self, pose_landmarks):
        """Calculate dynamic steering wheel box based on body pose landmarks"""

        wheel_box = self.default_wheel_box.copy()
        if pose_landmarks and pose_landmarks.pose_landmarks:
            landmarks = pose_landmarks.pose_landmarks.landmark
            
            if (landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5 and 
                landmarks[23].visibility > 0.5 and landmarks[24].visibility > 0.5):
                
                x11, y11 = landmarks[11].x, landmarks[11].y  # Left shoulder
                x12, y12 = landmarks[12].x, landmarks[12].y  # Right shoulder
                x23, y23 = landmarks[23].x, landmarks[23].y  # Left hip
                x24, y24 = landmarks[24].x, landmarks[24].y  # Right hip
                
                W = self.alpha * abs(x12 - x11)
                H = W * 0.6
                Cx = 0.5 * (x11 + x12)
                Cy = 0.5 * (y11 + y12) + self.beta * ((0.5 * (y23 + y24)) - 0.5 * (y11 + y12))
                
                xmin = max(0.0, Cx - W/2)
                xmax = min(1.0, Cx + W/2)
                ymin = max(0.0, Cy - 3*H/4)
                ymax = min(1.0, Cy + H/4)
                
                wheel_box = [xmin, xmax, ymin, ymax]
        
        self.current_wheel_box = wheel_box
        return wheel_box
    
    def extract_hand_position(self, pose_landmarks):
        hand_pos = {
            "left_wrist_x": 0, "left_wrist_y": 0, "left_wrist_z": 0,
            "left_palm_x": 0, "left_palm_y": 0, "left_palm_z": 0,
            "right_wrist_x": 0, "right_wrist_y": 0, "right_wrist_z": 0,
            "right_palm_x": 0, "right_palm_y": 0, "right_palm_z": 0
        }
        
        if not pose_landmarks.pose_landmarks:
            return hand_pos
        
        landmarks = pose_landmarks.pose_landmarks.landmark
        
        if landmarks[15].visibility > 0.5: 
            hand_pos["left_wrist_x"] = landmarks[15].x * self.frame_width
            hand_pos["left_wrist_y"] = landmarks[15].y * self.frame_height
            hand_pos["left_wrist_z"] = landmarks[15].z
            
            if landmarks[17].visibility > 0.5 and landmarks[19].visibility > 0.5:
                palm_x = (landmarks[17].x + landmarks[19].x) / 2
                palm_y = (landmarks[17].y + landmarks[19].y) / 2
                palm_z = (landmarks[17].z + landmarks[19].z) / 2
                
                hand_pos["left_palm_x"] = palm_x * self.frame_width
                hand_pos["left_palm_y"] = palm_y * self.frame_height
                hand_pos["left_palm_z"] = palm_z
        
        if landmarks[16].visibility > 0.5:  
            hand_pos["right_wrist_x"] = landmarks[16].x * self.frame_width
            hand_pos["right_wrist_y"] = landmarks[16].y * self.frame_height
            hand_pos["right_wrist_z"] = landmarks[16].z
            
            if landmarks[18].visibility > 0.5 and landmarks[20].visibility > 0.5:
                palm_x = (landmarks[18].x + landmarks[20].x) / 2
                palm_y = (landmarks[18].y + landmarks[20].y) / 2
                palm_z = (landmarks[18].z + landmarks[20].z) / 2
                
                hand_pos["right_palm_x"] = palm_x * self.frame_width
                hand_pos["right_palm_y"] = palm_y * self.frame_height
                hand_pos["right_palm_z"] = palm_z
                
        return hand_pos

    def compute_hand_eye_depth(self, pose_landmarks, face_landmarks):
        depths = {"left_hand_eye_depth": 0.0, "right_hand_eye_depth": 0.0}
        
        if not pose_landmarks.pose_landmarks or not face_landmarks.multi_face_landmarks:
            return depths
        
        eye_center = [0, 0, 0] 
        if face_landmarks.multi_face_landmarks:
            left_eye = face_landmarks.multi_face_landmarks[0].landmark[468]
            right_eye = face_landmarks.multi_face_landmarks[0].landmark[473]
            eye_center = [
                (left_eye.x + right_eye.x) / 2 * self.frame_width,
                (left_eye.y + right_eye.y) / 2 * self.frame_height,
                (left_eye.z + right_eye.z) / 2
            ]
        
        landmarks = pose_landmarks.pose_landmarks.landmark
        
        if landmarks[15].visibility > 0.5:  # Left wrist
            left_wrist_3d = [
                landmarks[15].x * self.frame_width,
                landmarks[15].y * self.frame_height,
                landmarks[15].z
            ]
            depths["left_hand_eye_depth"] = np.sqrt(sum((a - b) ** 2 for a, b in zip(left_wrist_3d, eye_center)))
        
        if landmarks[16].visibility > 0.5:  # Right wrist
            right_wrist_3d = [
                landmarks[16].x * self.frame_width,
                landmarks[16].y * self.frame_height,
                landmarks[16].z
            ]
            depths["right_hand_eye_depth"] = np.sqrt(sum((a - b) ** 2 for a, b in zip(right_wrist_3d, eye_center)))
            
        return depths

    def calculate_angle_3d(self, point1, point2, point3):
        """Calculate the angle between three 3D points, with point2 as the vertex"""

        vector1 = [point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]]
        vector2 = [point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]]
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        
        mag1 = np.sqrt(sum(a ** 2 for a in vector1))
        mag2 = np.sqrt(sum(a ** 2 for a in vector2))
        
        if mag1 * mag2 == 0:
            return 0.0
        
        cos_angle = dot_product / (mag1 * mag2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        return angle

    def extract_arm_angles(self, pose_landmarks):
        """Extract left and right elbow angles (shoulder-elbow-wrist)"""
        angles = {"left_elbow_angle": 0.0, "right_elbow_angle": 0.0}
        
        if not pose_landmarks.pose_landmarks:
            return angles
        
        landmarks = pose_landmarks.pose_landmarks.landmark
        
        if landmarks[12].visibility > 0.5 and landmarks[14].visibility > 0.5 and landmarks[16].visibility > 0.5:
            right_shoulder = [
                landmarks[12].x * self.frame_width,
                landmarks[12].y * self.frame_height,
                landmarks[12].z
            ]
            right_elbow = [
                landmarks[14].x * self.frame_width,
                landmarks[14].y * self.frame_height,
                landmarks[14].z
            ]
            right_wrist = [
                landmarks[16].x * self.frame_width,
                landmarks[16].y * self.frame_height,
                landmarks[16].z
            ]
            angles["right_elbow_angle"] = self.calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
        
        if landmarks[11].visibility > 0.5 and landmarks[13].visibility > 0.5 and landmarks[15].visibility > 0.5:
            left_shoulder = [
                landmarks[11].x * self.frame_width,
                landmarks[11].y * self.frame_height,
                landmarks[11].z
            ]
            left_elbow = [
                landmarks[13].x * self.frame_width,
                landmarks[13].y * self.frame_height,
                landmarks[13].z
            ]
            left_wrist = [
                landmarks[15].x * self.frame_width,
                landmarks[15].y * self.frame_height,
                landmarks[15].z
            ]
            angles["left_elbow_angle"] = self.calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
        
        return angles

    def check_hand_off_steering_wheel(self, pose_landmarks):
        current_wheel_box = self.calculate_wheel_box(pose_landmarks)
        
        result = {
            "left_hand_off_wheel": False,
            "left_hand_off_wheel_duration": sum(self.left_hand_off_wheel_history),
            "right_hand_off_wheel": False, 
            "right_hand_off_wheel_duration": sum(self.right_hand_off_wheel_history)
        }
        
        if not pose_landmarks.pose_landmarks:
            self.left_hand_off_wheel_history.append(1)
            self.right_hand_off_wheel_history.append(1)
            result["left_hand_off_wheel"] = True
            result["right_hand_off_wheel"] = True
            return result
        
        wheel_box_pixels = [
            current_wheel_box[0] * self.frame_width,  # x_min
            current_wheel_box[1] * self.frame_width,  # x_max
            current_wheel_box[2] * self.frame_height, # y_min
            current_wheel_box[3] * self.frame_height  # y_max
        ]
        
        landmarks = pose_landmarks.pose_landmarks.landmark
        
        if landmarks[15].visibility > 0.5:
            x, y = landmarks[15].x * self.frame_width, landmarks[15].y * self.frame_height
            
            in_wheel = (wheel_box_pixels[0] <= x <= wheel_box_pixels[1] and 
                        wheel_box_pixels[2] <= y <= wheel_box_pixels[3])
            
            if not in_wheel:
                result["left_hand_off_wheel"] = True
                self.left_hand_off_wheel_history.append(1)
            else:
                self.left_hand_off_wheel_history.append(0)
        else:
            self.left_hand_off_wheel_history.append(1)
            result["left_hand_off_wheel"] = True
            
        if landmarks[16].visibility > 0.5:
            x, y = landmarks[16].x * self.frame_width, landmarks[16].y * self.frame_height
            
            in_wheel = (wheel_box_pixels[0] <= x <= wheel_box_pixels[1] and 
                        wheel_box_pixels[2] <= y <= wheel_box_pixels[3])
            
            if not in_wheel:
                result["right_hand_off_wheel"] = True
                self.right_hand_off_wheel_history.append(1)
            else:
                self.right_hand_off_wheel_history.append(0)
        else:
            self.right_hand_off_wheel_history.append(1)
            result["right_hand_off_wheel"] = True
            
        result["left_hand_off_wheel_duration"] = sum(self.left_hand_off_wheel_history)
        result["right_hand_off_wheel_duration"] = sum(self.right_hand_off_wheel_history)
        
        return result
    
    def check_hand_proximity_to_face(self, pose_landmarks, face_landmarks):
            result = {
                "left_hand_distance": 0,
                "right_hand_distance": 0
            }
            
            if not pose_landmarks.pose_landmarks or not face_landmarks.multi_face_landmarks:
                self.left_hand_near_face_history.append(0)
                self.right_hand_near_face_history.append(0)
                return result
            
            nose = face_landmarks.multi_face_landmarks[0].landmark[1]  # Nose tip landmark
            nose_pos = [nose.x * self.frame_width, nose.y * self.frame_height]
            
            landmarks = pose_landmarks.pose_landmarks.landmark
            
            if landmarks[15].visibility > 0.5:
                wrist_pos = [landmarks[15].x * self.frame_width, landmarks[15].y * self.frame_height]
                
                distance = np.sqrt((wrist_pos[0] - nose_pos[0])**2 + (wrist_pos[1] - nose_pos[1])**2)
                result["left_hand_distance"] = distance
                
                self.left_hand_near_face_history.append(distance)
            else:
                self.left_hand_near_face_history.append(0)
                
            if landmarks[16].visibility > 0.5:
                wrist_pos = [landmarks[16].x * self.frame_width, landmarks[16].y * self.frame_height]
                
                distance = np.sqrt((wrist_pos[0] - nose_pos[0])**2 + (wrist_pos[1] - nose_pos[1])**2)
                result["right_hand_distance"] = distance
                
                self.right_hand_near_face_history.append(distance)
            else:
                self.right_hand_near_face_history.append(0)
            
            left_distances = [d for d in self.left_hand_near_face_history if d > 0]
            right_distances = [d for d in self.right_hand_near_face_history if d > 0]
            
            result["left_hand_distance_var"] = np.var(left_distances) if len(left_distances) > 1 else 0.0
            result["right_hand_distance_var"] = np.var(right_distances) if len(right_distances) > 1 else 0.0
            
            return result

    def calculate_hand_features(self, pose_results, face_landmarks, frame_id):
        current_wheel_box = self.calculate_wheel_box(pose_results)
        
        hand_pos = self.extract_hand_position(pose_results)
        hand_eye_depth = self.compute_hand_eye_depth(pose_results, face_landmarks)
        arm_angles = self.extract_arm_angles(pose_results)
        hand_off_wheel = self.check_hand_off_steering_wheel(pose_results)
        hand_proximity = self.check_hand_proximity_to_face(pose_results, face_landmarks)
        
        wheel_box_info = {
            "wheel_box_x_min": current_wheel_box[0],
            "wheel_box_x_max": current_wheel_box[1],
            "wheel_box_y_min": current_wheel_box[2],
            "wheel_box_y_max": current_wheel_box[3]
        }
        
        data = {
            **hand_pos,
            **hand_eye_depth,
            **arm_angles,
            **hand_off_wheel,
            **hand_proximity
        }
        
        return data


def process_video_hand(video_path, proximity_threshold=0.2, 
                       alpha=1.8, beta=1.5, 
                       default_wheel_box=[0.25, 0.92, 0.4, 0.6], 
                       window_size=150):
    df = pd.DataFrame()
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    hand_feature = hand_features(
        frame_width=frame_width, 
        frame_height=frame_height,
        proximity_threshold=proximity_threshold,
        alpha=alpha,
        beta=beta,
        default_wheel_box=default_wheel_box,
        window_size=window_size
    )
    
    frame_id = 0
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    mp_pose = mp.solutions.pose
    pose_mesh = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        face_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_results = pose_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        features = hand_feature.calculate_hand_features(pose_results, face_results, frame_id)
        
        df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
        
        frame_id += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    return df

if __name__ == "__main__":
    df = process_video_hand(
        video_path="/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/normal_5.mp4",
        proximity_threshold=0.2,
        alpha=1.8,
        beta=1.5,
        default_wheel_box=[0.25, 0.92, 0.4, 0.6],
        window_size=150
    )
    print(df.head())