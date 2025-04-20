import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from collections import deque

class Posepoints():
    def __init__(self, face_mesh_object, hands_object, df=None, frame_width=640, frame_height=480, max_side_glance=60, mar_threshold = 0.5, perclos_threshold=0.18):
        self.face_mesh_object = face_mesh_object
        self.hands_object = hands_object
        self.df = df if df is not None else pd.DataFrame()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_side_glance = max_side_glance
        self.mar_threshold= mar_threshold
        self.ear_window = 30 # 1 sec @ 30fps
        self.perclos_window = 60
        self.perclos_threshold = perclos_threshold
        
    
    def process_frame(self, frame, frame_id):
        face_results = self.face_mesh_object.process(frame)
        hand_results = self.hands_object.process(frame)
        self.frame_id = frame_id
        self.save_to_df(face_results, hand_results)
        return face_results, hand_results
    
    @staticmethod
    def get_mouth_landmarks(face_results, frame_width=640, frame_height=480):
        mouth_landmarks =[]
        if face_results.multi_face_landmarks:
            for face_landmakrs in face_results.multi_face_landmarks:
                mouth_landmarks = [
                    face_landmakrs.landmark[67], # Right Corner of LiP
                    face_landmakrs.landmark[73], # Right top of LiP
                    face_landmakrs.landmark[11], # Top of LIP
                    face_landmakrs.landmark[303], # Left top of LiP
                    face_landmakrs.landmark[61], # Left Corner  of Lip
                    face_landmakrs.landmark[403], # Left Botttom of LiP
                    face_landmakrs.landmark[16], # Botttom of LiP
                    face_landmakrs.landmark[180], # Right Botttom of LiP
                ]
                mouth_landmarks = [(lm.x * frame_width, lm.y*frame_height, lm.z) for lm in mouth_landmarks]
                return mouth_landmarks

    @staticmethod
    def get_eye_landmarks(face_results, frame_width=640, frame_height=480):
        left_eye_landmarks = []
        right_eye_landmarks = []
        left_pupil_landmarks = []
        right_pupil_landmarks = []

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                left_eye_landmarks = [
                    face_landmarks.landmark[33], #left most point # p1
                    face_landmarks.landmark[133], # right most point # p4
                    face_landmarks.landmark[160], # top left point # p2
                    face_landmarks.landmark[158], # top right point # p3
                    face_landmarks.landmark[144], # bottom left point # p6
                    face_landmarks.landmark[153], # bottom right point # p5
                ]

                right_eye_landmarks = [
                    face_landmarks.landmark[362], # left most point
                    face_landmarks.landmark[263], # right most point
                    face_landmarks.landmark[385], # top left point
                    face_landmarks.landmark[387], # top right point
                    face_landmarks.landmark[380], # bottom left point
                    face_landmarks.landmark[373], # bottom right point
                ]

                left_pupil_landmarks = [
                    face_landmarks.landmark[469], # right most point
                    face_landmarks.landmark[470], # top point
                    face_landmarks.landmark[471], # left most point
                    face_landmarks.landmark[472], # bottom point
                ]

                right_pupil_landmarks = [
                    face_landmarks.landmark[474], # right most point
                    face_landmarks.landmark[475], # top point
                    face_landmarks.landmark[476], # left most point
                    face_landmarks.landmark[477], # bottom point
                ]
                # Convert landmarks to list
                left_eye_landmarks = [(landmark.x * frame_width, landmark.y * frame_height, landmark.z) for landmark in left_eye_landmarks]
                right_eye_landmarks = [(landmark.x * frame_width, landmark.y * frame_height, landmark.z) for landmark in right_eye_landmarks]
                left_pupil_landmarks = [(landmark.x * frame_width, landmark.y * frame_height, landmark.z) for landmark in left_pupil_landmarks]
                right_pupil_landmarks = [(landmark.x * frame_width, landmark.y * frame_height, landmark.z) for landmark in right_pupil_landmarks]

        return left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks
    
    def get_features(self, face_results, hand_results):
        feature_points = {
            "left_eye": [],
            "right_eye": [],
            "left_pupil": [],
            "right_pupil": [],
            "mouth" : [],
            "frame_id": self.frame_id,
        }

        feature_points["left_eye"], feature_points["right_eye"], feature_points["left_pupil"], feature_points["right_pupil"] = self.get_eye_landmarks(face_results, self.frame_width, self.frame_height)
        feature_points["mouth"] = self.get_mouth_landmarks(face_results, self.frame_width, self.frame_height)
            
        return feature_points
    
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
        #print(point1, point2)
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @staticmethod
    def euclidean_distance_3d(point1, point2):
        """Calculates the Euclidean distance between two 3D points."""
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    @staticmethod
    def MouthAspectRatio(mouth):
        if len(mouth) <4:
            return 0.0
        p1 = mouth[0]
        p2 = mouth[1]
        p3 = mouth[2]
        p4 = mouth[3]
        p5 = mouth[4]
        p6 = mouth[5]
        p7 = mouth[7]
        p8 = mouth[8]

        horizontal_dist = Posepoints.euclidean_distance_2d(p1, p6)

        if horizontal_dist < 1e-5:
            return 0.0
        
        mar = (Posepoints.euclidean_distance_2d(p2,p8) + Posepoints.euclidean_distance_2d(p3, p7) + Posepoints.euclidean_distance_2d(p4,p6))/(2*Posepoints.euclidean_distance_2d(p1,p5))
        return mar
    
    def calculate_perclos(self):
        """Calculate perclos
        Calculate EAR per frame. Given an EAR threshold for "eye closure" i.e. self.perclos_threshold. Maintain a rolling window of frames given in self .perclos_window. Calculate the percentage of frames within the window where EAR < threshold. 
        """
        if len(self.df)<self.perclose_window:
            return 0.0
        
        window = self.df. tail(self.perclos_threshold)
        valid_frames = window[~window['no_visible_eyes']]
        if len(valid_frames) == 0:
            return 0.0
        avg_ear = (valid_frames['left_eye_aspect_ratio'] + 
          valid_frames['right_eye_aspect_ratio']) / 2
        closed_frames = avg_ear < self.perclos_threshold
        return closed_frames.mean()
    
    @staticmethod
    def EyeAspectRatio2D(eye):
        #print(eye)
        # Formula : dist(p2,p6) + dist(p3,p5) / 2*dist(p1, p4)
        if eye is None or len(eye) < 6:
            return 0
        p1 = (eye[0][0], eye[0][1], eye[0][2])
        p2 = (eye[2][0], eye[2][1], eye[2][2])
        p3 = (eye[3][0], eye[3][1], eye[3][2])
        p4 = (eye[1][0], eye[1][1], eye[1][2])
        p5 = (eye[5][0], eye[5][1], eye[5][2])
        p6 = (eye[4][0], eye[4][1], eye[4][2])

        dist_p2_p6 = Posepoints.euclidean_distance_2d(p2, p6)
        dist_p3_p5 = Posepoints.euclidean_distance_2d(p3, p5)
        dist_p1_p4 = Posepoints.euclidean_distance_2d(p1, p4)

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

        dist_p2_p6 = Posepoints.euclidean_distance_3d(p2, p6)
        dist_p3_p5 = Posepoints.euclidean_distance_3d(p3, p5)
        dist_p1_p4 = Posepoints.euclidean_distance_3d(p1, p4)

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
        
        eye_center = Posepoints.center_eye(eye)
        pupil_center = Posepoints.center_pupil(pupil)

        # add a zero in the z axis
        eye_center = (eye_center[0], eye_center[1], 0)
        pupil_center = (pupil_center[0], pupil_center[1], 0)
        
        if eye_center is None or pupil_center is None:
            return None
        
        return Posepoints.euclidean_distance_2d(eye_center, pupil_center)
    
    def count_blinks_in_window(self, num_frames=30):
        if self.df.empty or len(self.df) < 3:  # Need at least 3 frames to detect a complete blink
            return 0
            
        available_frames = min(len(self.df), num_frames)
        window_data = self.df.tail(available_frames).copy()
        
        if len(window_data) < 3:
            return 0
        
        window_data['both_eyes_closed'] = (window_data['left_eye_closed'] & 
                                        window_data['right_eye_closed']).astype(int)
        
        window_data['state_change'] = window_data['both_eyes_closed'].diff().fillna(0)
        
       
        blink_count = len(window_data[window_data['state_change'] == -1])
        
        return blink_count
    
    def variance_pupil_movement(self, eye_pupil_distance, side="left", num_frames=10):
        """Calculates the variance of pupil movement over a specified number of frames."""
        if self.df.empty:
            return 0, 0
        if side not in ["left", "right"]:
            raise ValueError("Side must be either 'left' or 'right'")
        if len(self.df) < num_frames:
            return 0, 0
        last_rows = self.df.tail(num_frames)
        if side == "left":
            last_rows = last_rows[last_rows["left_eye_pupil_distance"].notna()]
            pupil_distance = last_rows["left_eye_pupil_distance"].values
        else:
            last_rows = last_rows[last_rows["right_eye_pupil_distance"].notna()]
            pupil_distance = last_rows["right_eye_pupil_distance"].values
        if len(pupil_distance) < 2:
            return 0
        mean_distance = np.mean(pupil_distance)
        variance = np.var(pupil_distance)
        return mean_distance, variance

    def calculate_ear_variance(self):
        """Calculate mean and variance of eye aspect ratio"""
        if len(self.df) < self.ear_window_size:
            return 0.0, 0.0
        
        # We get a more human meaningful measure-reduce noise
        avg_ears = (
            self.df['left_eye_aspect_ratio'].tail(self.ear_window_size) + 
            self.df['right_eye_aspect_ratio'].tail(self.ear_window_size)
        ) / 2
        
        return np.mean(avg_ears), np.var(avg_ears.values)

    @staticmethod
    def is_eye_closed(eye, threshold=0.2):
        """Checks if the eye is closed based on the aspect ratio."""
        aspect_ratio = Posepoints.EyeAspectRatio2D(eye)
        return aspect_ratio < threshold
    
    def calculate_eye_features(self, face_results, hand_results):
        feature_points = self.get_features(face_results, hand_results)
        frame_id = self.frame_id
        left_eye = feature_points["left_eye"]
        right_eye = feature_points["right_eye"]
        left_pupil = feature_points["left_pupil"]
        right_pupil = feature_points["right_pupil"]
        mouth = feature_points['mouth']

        # Eye based metric
        no_visible_eyes = self.no_visible_eyes(face_results)
        left_eye_aspect_ratio = self.EyeAspectRatio2D(left_eye)
        right_eye_aspect_ratio = self.EyeAspectRatio2D(right_eye)
        left_eye_aspect_ratio_3d = self.EyeAspectRatio3D(left_eye)
        right_eye_aspect_ratio_3d = self.EyeAspectRatio3D(right_eye)
        left_eye_pupil_distance = self.eye_pupling_distance(left_eye, left_pupil)
        right_eye_pupil_distance = self.eye_pupling_distance(right_eye, right_pupil)
        

        left_eye_pupil_movement, left_eye_pupil_variance = self.variance_pupil_movement(left_eye_pupil_distance, side="left")
        right_eye_pupil_movement, right_eye_pupil_variance = self.variance_pupil_movement(right_eye_pupil_distance, side="right")
        left_eye_closed = self.is_eye_closed(left_eye)
        right_eye_closed = self.is_eye_closed(right_eye)
        num_blinks = self.count_blinks_in_window(num_frames=30)
        ear_variance = self.calculate_ear_variance()

        perclos = self.calculate_perclos()
        # Mouth based metrics
        mouth_aspect_ratio = self.MouthAspectRatio(mouth)
        # Mouth and eye based metric
        eye_closure_during_yawn = (mouth_aspect_ratio > self.mar_threshold) and (left_eye_closed and right_eye_closed)
        
        
        data = {
            "frame_id": frame_id,
            "left_eye_aspect_ratio": left_eye_aspect_ratio,
            "right_eye_aspect_ratio": right_eye_aspect_ratio,
            "left_eye_aspect_ratio_3d": left_eye_aspect_ratio_3d,
            "right_eye_aspect_ratio_3d": right_eye_aspect_ratio_3d,
            "left_eye_pupil_distance": left_eye_pupil_distance,
            "right_eye_pupil_distance": right_eye_pupil_distance,
            "no_visible_eyes": no_visible_eyes,
            "left_eye_pupil_movement": left_eye_pupil_movement,
            "left_eye_pupil_variance": left_eye_pupil_variance,
            "right_eye_pupil_movement": right_eye_pupil_movement,
            "right_eye_pupil_variance": right_eye_pupil_variance,
            "left_eye_closed": left_eye_closed,
            "right_eye_closed": right_eye_closed,
            "num_blinks": num_blinks,
            "mouth_aspect_ratio": mouth_aspect_ratio,
            "eye_closure_during_yawn": eye_closure_during_yawn,
            'ear_variance': ear_variance,
            "perclos": perclos,
        }

        return data




    
    def save_to_df(self, face_results, hand_results):

        eye_data = self.calculate_eye_features(face_results, hand_results)
        data = eye_data
        # it will merge all the incoming data from eye, hand, face, etc
        frame_id = data["frame_id"]     

        

        self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
        print(f"Frame {frame_id} data added to internal DataFrame.")
    
    def save_to_csv(self, csv_path):
        self.csv_path = csv_path
        self.df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}.")


if __name__ == "__main__":
    # Example usage
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands()
    
    cap = cv2.VideoCapture("/home/harsh/Downloads/sem2/edgeai/edge ai project/video5.mp4")

    frame_id = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('eye_function.avi', fourcc, fps_input, (frame_width, frame_height))

    extractor = Posepoints(face_mesh, hands, frame_width=frame_width, frame_height=frame_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_id += 1
        face_results, hand_results = extractor.process_frame(frame_rgb, frame_id)
        
        out.write(frame)
        
        # Optional: Show the frame
        # cv2.imshow('MediaPipe Face Mesh', frame)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
            
    extractor.save_to_csv("eye_data.csv")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

        






