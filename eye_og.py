import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from collections import deque

class eye_features():
    def __init__(self, frame_width, frame_height, mar_threshold, perclos_threshold):
        self.mar_threshold = mar_threshold
        self.perclos_threshold = perclos_threshold
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Window sizes
        self.ear_window_size = 150  # 5 sec @ 30fps
        self.perclos_window_size = 150
        self.blink_window_size = 150
        self.pupil_movement_window_size = 30
        
        # Initialize windows using deques
        self.perclos_window = deque(maxlen=self.perclos_window_size)
        self.blink_window = deque(maxlen=self.blink_window_size)
        self.ear_window = deque(maxlen=self.ear_window_size)
        self.left_pupil_distance_window = deque(maxlen=self.pupil_movement_window_size)
        self.right_pupil_distance_window = deque(maxlen=self.pupil_movement_window_size)

    @staticmethod
    def get_mouth_landmarks(face_results, frame_width=640, frame_height=480):
        mouth_landmarks = []
        if face_results.multi_face_landmarks:
            for face_landmakrs in face_results.multi_face_landmarks:
                mouth_landmarks = [
                    face_landmakrs.landmark[67],  # Right Corner of LiP
                    face_landmakrs.landmark[73],  # Right top of LiP
                    face_landmakrs.landmark[11],  # Top of LIP
                    face_landmakrs.landmark[303],  # Left top of LiP
                    face_landmakrs.landmark[61],  # Left Corner of Lip
                    face_landmakrs.landmark[403],  # Left Bottom of LiP
                    face_landmakrs.landmark[16],  # Bottom of LiP
                    face_landmakrs.landmark[180],  # Right Bottom of LiP
                ]
                mouth_landmarks = [(lm.x * frame_width, lm.y * frame_height, lm.z) for lm in mouth_landmarks]
                return mouth_landmarks
        return None

    @staticmethod
    def get_eye_landmarks(face_results, frame_width=640, frame_height=480):
        left_eye_landmarks = []
        right_eye_landmarks = []
        left_pupil_landmarks = []
        right_pupil_landmarks = []

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                left_eye_landmarks = [
                    face_landmarks.landmark[33],  # left most point # p1
                    face_landmarks.landmark[133],  # right most point # p4
                    face_landmarks.landmark[160],  # top left point # p2
                    face_landmarks.landmark[158],  # top right point # p3
                    face_landmarks.landmark[144],  # bottom left point # p6
                    face_landmarks.landmark[153],  # bottom right point # p5
                ]

                right_eye_landmarks = [
                    face_landmarks.landmark[362],  # left most point
                    face_landmarks.landmark[263],  # right most point
                    face_landmarks.landmark[385],  # top left point
                    face_landmarks.landmark[387],  # top right point
                    face_landmarks.landmark[380],  # bottom left point
                    face_landmarks.landmark[373],  # bottom right point
                ]

                left_pupil_landmarks = [
                    face_landmarks.landmark[469],  # right most point
                    face_landmarks.landmark[470],  # top point
                    face_landmarks.landmark[471],  # left most point
                    face_landmarks.landmark[472],  # bottom point
                ]

                right_pupil_landmarks = [
                    face_landmarks.landmark[474],  # right most point
                    face_landmarks.landmark[475],  # top point
                    face_landmarks.landmark[476],  # left most point
                    face_landmarks.landmark[477],  # bottom point
                ]
                # Convert landmarks to list
                left_eye_landmarks = [(landmark.x * frame_width, landmark.y * frame_height, landmark.z) for landmark in
                                     left_eye_landmarks]
                right_eye_landmarks = [(landmark.x * frame_width, landmark.y * frame_height, landmark.z) for landmark in
                                      right_eye_landmarks]
                left_pupil_landmarks = [(landmark.x * frame_width, landmark.y * frame_height, landmark.z) for landmark in
                                       left_pupil_landmarks]
                right_pupil_landmarks = [(landmark.x * frame_width, landmark.y * frame_height, landmark.z) for landmark in
                                        right_pupil_landmarks]

        return left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks

    @staticmethod
    def no_visible_eyes(face_results):
        if not face_results.multi_face_landmarks:
            return True
        for face_landmarks in face_results.multi_face_landmarks:
            left_eye_visible = face_landmarks.landmark[33].visibility > 0.5
            right_eye_visible = face_landmarks.landmark[362].visibility > 0.5
            if left_eye_visible and right_eye_visible:
                return False
        return True

    @staticmethod
    def euclidean_distance_2d(point1, point2):
        """Calculates the Euclidean distance between two 2D points."""
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def euclidean_distance_3d(point1, point2):
        """Calculates the Euclidean distance between two 3D points."""
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    @staticmethod
    def MouthAspectRatio(mouth):
        if mouth is None or len(mouth) < 8:
            return 0.0
        p1 = mouth[0]
        p2 = mouth[1]
        p3 = mouth[2]
        p4 = mouth[3]
        p5 = mouth[4]
        p6 = mouth[5]
        p7 = mouth[6]
        p8 = mouth[7]

        horizontal_dist = eye_features.euclidean_distance_2d(p1, p6)

        if horizontal_dist < 1e-5:
            return 0.0

        mar = (eye_features.euclidean_distance_2d(p2, p8) + eye_features.euclidean_distance_2d(p3, p7) +
              eye_features.euclidean_distance_2d(p4, p6)) / (2 * eye_features.euclidean_distance_2d(p1, p5))
        return mar

    def update_perclos_window(self, frame_data):
        """Add a frame's eye data to the PERCLOS window"""
        self.perclos_window.append(frame_data)

    def calculate_perclos(self):
        """Calculate PERCLOS (percentage of eye closure)"""
        if len(self.perclos_window) < 1:
            return 0.0

        valid_frames = [frame for frame in self.perclos_window if not frame['no_visible_eyes']]
        if len(valid_frames) == 0:
            return 0.0

        avg_ear = [(frame['left_eye_aspect_ratio'] + frame['right_eye_aspect_ratio']) / 2 for frame in valid_frames]
        closed_frames = [ear for ear in avg_ear if ear < self.perclos_threshold]
        return len(closed_frames) / len(valid_frames)

    @staticmethod
    def EyeAspectRatio2D(eye):
        # Formula : dist(p2,p6) + dist(p3,p5) / 2*dist(p1, p4)
        if eye is None or len(eye) < 6:
            return 0
        p1 = (eye[0][0], eye[0][1], eye[0][2])
        p2 = (eye[2][0], eye[2][1], eye[2][2])
        p3 = (eye[3][0], eye[3][1], eye[3][2])
        p4 = (eye[1][0], eye[1][1], eye[1][2])
        p5 = (eye[5][0], eye[5][1], eye[5][2])
        p6 = (eye[4][0], eye[4][1], eye[4][2])

        dist_p2_p6 = eye_features.euclidean_distance_2d(p2, p6)
        dist_p3_p5 = eye_features.euclidean_distance_2d(p3, p5)
        dist_p1_p4 = eye_features.euclidean_distance_2d(p1, p4)

        if dist_p1_p4 == 0:
            return 0  # Avoid division by zero

        return (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)

    @staticmethod
    def EyeAspectRatio3D(eye):
        # Formula : dist(p2,p6) + dist(p3,p5) / 2*dist(p1, p4)
        if eye is None or len(eye) < 6:
            return 0
        p1 = (eye[0][0], eye[0][1], eye[0][2])
        p2 = (eye[2][0], eye[2][1], eye[2][2])
        p3 = (eye[3][0], eye[3][1], eye[3][2])
        p4 = (eye[1][0], eye[1][1], eye[1][2])
        p5 = (eye[5][0], eye[5][1], eye[5][2])
        p6 = (eye[4][0], eye[4][1], eye[4][2])

        dist_p2_p6 = eye_features.euclidean_distance_3d(p2, p6)
        dist_p3_p5 = eye_features.euclidean_distance_3d(p3, p5)
        dist_p1_p4 = eye_features.euclidean_distance_3d(p1, p4)

        if dist_p1_p4 == 0:
            return 0  # Avoid division by zero

        return (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)

    @staticmethod
    def center_pupil(pupil):
        """Calculates the center of the pupil given its landmarks."""
        if not pupil or len(pupil) < 4:
            return 0

        x_coords = [p[0] for p in pupil]
        y_coords = [p[1] for p in pupil]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (center_x, center_y)

    @staticmethod
    def center_eye(eye):
        """Calculates the center of the eye given its landmarks."""
        if not eye or len(eye) < 6:
            return 0

        x_coords = [p[0] for p in eye]
        y_coords = [p[1] for p in eye]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (center_x, center_y)

    @staticmethod
    def eye_pupling_distance(eye, pupil):
        """Calculates the distance between the eye and pupil centers."""
        if eye is None or pupil is None:
            return 0
        if len(eye) < 6 or len(pupil) < 4:
            return 0

        eye_center = eye_features.center_eye(eye)
        pupil_center = eye_features.center_pupil(pupil)

        # add a zero in the z axis
        eye_center = (eye_center[0], eye_center[1], 0)
        pupil_center = (pupil_center[0], pupil_center[1], 0)

        if eye_center is None or pupil_center is None:
            return None

        return eye_features.euclidean_distance_2d(eye_center, pupil_center)

    def update_blink_window(self, frame_data):
        """Add a frame's eye data to the blink window"""
        self.blink_window.append(frame_data)
        
    def count_blinks_in_window(self):
        """Count the number of blinks in the current window"""
        if len(self.blink_window) < 3:  # Need at least 3 frames to detect a complete blink
            return 0

        window_data = list(self.blink_window)

        if len(window_data) < 3:
            return 0

        both_eyes_closed = [(frame['left_eye_closed'] & frame['right_eye_closed']) for frame in window_data]
        def count_blinks(both_eyes_closed):
            blinks = 0
            for i in range(1, len(both_eyes_closed) - 1):
                if both_eyes_closed[i-1] != both_eyes_closed[i] and both_eyes_closed[i+1] == both_eyes_closed[i-1]:
                    blinks += 1
            return blinks
        blink_count = count_blinks(both_eyes_closed)

        return blink_count

    def update_pupil_distance_windows(self, left_distance, right_distance):
        """Add pupil distances to the respective windows"""
        if left_distance is not None:
            self.left_pupil_distance_window.append(left_distance)
        if right_distance is not None:
            self.right_pupil_distance_window.append(right_distance)

    def variance_pupil_movement(self, side="left"):
        """Calculates the variance of pupil movement over the window."""
        if side not in ["left", "right"]:
            raise ValueError("Side must be either 'left' or 'right'")
            
        if side == "left":
            distances = list(self.left_pupil_distance_window)
        else:
            distances = list(self.right_pupil_distance_window)
            
        if len(distances) < 2:
            return 0, 0
            
        mean_distance = np.mean(distances)
        variance = np.var(distances)
        return mean_distance, variance

    def update_ear_window(self, left_ear, right_ear):
        """Add EAR values to the window"""
        self.ear_window.append((left_ear + right_ear) / 2)  # Store the average EAR

    def calculate_ear_variance(self):
        """Calculate mean and variance of eye aspect ratio"""
        if len(self.ear_window) < 2:
            return 0.0, 0.0
        
        avg_ears = list(self.ear_window)
        return np.mean(avg_ears), np.var(avg_ears)

    @staticmethod
    def is_eye_closed(eye, threshold=0.2):
        """Checks if the eye is closed based on the aspect ratio."""
        aspect_ratio = eye_features.EyeAspectRatio2D(eye)
        return aspect_ratio < threshold

    def calculate_eye_features(self, face_results, frame_id):
        """Calculate all eye features for the current frame and update all windows"""
        feature_points = self.get_features_eye(face_results, frame_id)
        
        left_eye = feature_points["left_eye"]
        right_eye = feature_points["right_eye"]
        left_pupil = feature_points["left_pupil"]
        right_pupil = feature_points["right_pupil"]
        mouth = feature_points['mouth']

        # Eye based metrics
        no_visible_eyes = self.no_visible_eyes(face_results)
        left_eye_aspect_ratio = self.EyeAspectRatio2D(left_eye)
        right_eye_aspect_ratio = self.EyeAspectRatio2D(right_eye)
        left_eye_aspect_ratio_3d = self.EyeAspectRatio3D(left_eye)
        right_eye_aspect_ratio_3d = self.EyeAspectRatio3D(right_eye)
        left_eye_pupil_distance = self.eye_pupling_distance(left_eye, left_pupil)
        right_eye_pupil_distance = self.eye_pupling_distance(right_eye, right_pupil)
        left_eye_closed = self.is_eye_closed(left_eye)
        right_eye_closed = self.is_eye_closed(right_eye)
        
        # Mouth based metrics
        mouth_aspect_ratio = self.MouthAspectRatio(mouth)
        
        # Mouth and eye based metric
        eye_closure_during_yawn = (mouth_aspect_ratio > self.mar_threshold) and (left_eye_closed and right_eye_closed)
        
        # Create frame data dictionary
        frame_data = {
            "frame_id": frame_id,
            "left_eye_aspect_ratio": left_eye_aspect_ratio,
            "right_eye_aspect_ratio": right_eye_aspect_ratio,
            "left_eye_aspect_ratio_3d": left_eye_aspect_ratio_3d,
            "right_eye_aspect_ratio_3d": right_eye_aspect_ratio_3d,
            "left_eye_pupil_distance": left_eye_pupil_distance,
            "right_eye_pupil_distance": right_eye_pupil_distance,
            "no_visible_eyes": no_visible_eyes,
            "left_eye_closed": left_eye_closed,
            "right_eye_closed": right_eye_closed,
            "mouth_aspect_ratio": mouth_aspect_ratio,
            "eye_closure_during_yawn": eye_closure_during_yawn,
        }
        
        # Update all the windows with this frame's data
        self.update_perclos_window(frame_data)
        self.update_blink_window(frame_data)
        self.update_ear_window(left_eye_aspect_ratio, right_eye_aspect_ratio)
        self.update_pupil_distance_windows(left_eye_pupil_distance, right_eye_pupil_distance)
        
        # Calculate window-based metrics
        left_eye_pupil_movement, left_eye_pupil_variance = self.variance_pupil_movement(side="left")
        right_eye_pupil_movement, right_eye_pupil_variance = self.variance_pupil_movement(side="right")
        num_blinks = self.count_blinks_in_window()
        ear_mean, ear_variance = self.calculate_ear_variance()
        perclos = self.calculate_perclos()
        
        # Add window-based metrics to the frame data
        frame_data.update({
            "left_eye_pupil_movement": left_eye_pupil_movement,
            "left_eye_pupil_variance": left_eye_pupil_variance,
            "right_eye_pupil_movement": right_eye_pupil_movement,
            "right_eye_pupil_variance": right_eye_pupil_variance,
            "num_blinks": num_blinks,
            "ear_mean": ear_mean,
            "ear_variance": ear_variance,
            "perclos": perclos,
        })
        
        return frame_data

    def get_features_eye(self, face_results, frame_id):
        feature_points = {
            "left_eye": [],
            "right_eye": [],
            "left_pupil": [],
            "right_pupil": [],
            "mouth": [],
            "frame_id": frame_id,
        }

        feature_points["left_eye"], feature_points["right_eye"], feature_points["left_pupil"], feature_points["right_pupil"] = self.get_eye_landmarks(
            face_results, self.frame_width, self.frame_height)
        feature_points["mouth"] = self.get_mouth_landmarks(face_results, self.frame_width, self.frame_height)

        return feature_points





def process_video(video_path, mar_threshold=0.2, perclos_threshold=0.2):
    df = pd.DataFrame()
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eye_feature = eye_features(frame_width, frame_height, mar_threshold, perclos_threshold)
    frame_id = 0

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #print(frame_id)
        # Process frame with both eye feature classes
        face_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        features = eye_feature.calculate_eye_features(face_results, frame_id=frame_id)

        df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
        frame_id += 1
    cap.release()
    cv2.destroyAllWindows()

    return df


if __name__ == "__main__":
    # Example usage
    video_path = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/normal_5.mp4"
    mar_threshold = 0.2
    perclos_threshold = 0.2
    
    # Process the video and get the features DataFrame
    df = process_video(video_path, mar_threshold, perclos_threshold)
    
    # Save the results to CSV
    output_csv = "eye_data.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"Processing complete. Data saved to {output_csv}")
    
    # Optional: Create an annotated video
    # cap = cv2.VideoCapture(video_path)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('eye_function.avi', fourcc, fps_input, (frame_width, frame_height))
    # # (Add video processing with annotations here if needed)
    # cap.release()
    # out.release()

        






