import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import threading
import queue
from collections import deque
import time

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from collections import deque

class RunningStat:
    def __init__(self, window_size):
        self.n = window_size
        self.count = 0
        self.mean = 0.0
        self.var = 0.0   # stores E[x²] – (E[x])²
        self._buffer = deque(maxlen=window_size)
        
    def update(self, new_value):
        """Push a new value; if buffer is full, pop the oldest."""
        if self.count < self.n:
            # filling phase
            old = 0.0
            self.count += 1
        else:
            old = self._buffer[0]  # oldest about to be removed
            
        # push new, pop old
        if len(self._buffer) == self.n:
            self._buffer.popleft()
        self._buffer.append(new_value)
        
        # incremental mean
        self.mean += (new_value - old) / self.n
        # incremental E[x^2]
        self.var += (new_value*new_value - old*old) / self.n
        
    @property
    def variance(self):
        """Returns Var = E[x²] – (E[x])²"""
        return max(self.var - self.mean*self.mean, 0.0)
        
    @property
    def is_full(self):
        return self.count >= self.n


class BlinkDetector:
    """Specialized class for blink detection"""
    def __init__(self, window_size=150):
        self.window_size = window_size
        self.blink_window = deque(maxlen=window_size)
        
    def update(self, left_eye_closed, right_eye_closed):
        self.blink_window.append((left_eye_closed, right_eye_closed))
        
    def count_blinks(self):
        if len(self.blink_window) < 3:
            return 0
            
        window_data = list(self.blink_window)
        both_eyes_closed = [(left & right) for left, right in window_data]
        
        blinks = 0
        for i in range(1, len(both_eyes_closed) - 1):
            if both_eyes_closed[i-1] != both_eyes_closed[i] and both_eyes_closed[i+1] == both_eyes_closed[i-1]:
                blinks += 1
                
        return blinks

class PerclosCalculator:
    """Specialized class for PERCLOS calculation"""
    def __init__(self, window_size=150, threshold=0.2):
        self.window_size = window_size
        self.threshold = threshold
        self.perclos_window = deque(maxlen=window_size)
        
    def update(self, left_ear, right_ear, no_visible_eyes):
        self.perclos_window.append((left_ear, right_ear, no_visible_eyes))
        
    def calculate(self):
        if not self.perclos_window:
            return 0.0
            
        valid_frames = [(left, right) for left, right, no_visible in self.perclos_window if not no_visible]
        if not valid_frames:
            return 0.0
            
        avg_ears = [(left + right) / 2 for left, right in valid_frames]
        closed_frames = [ear for ear in avg_ears if ear < self.threshold]
        
        return len(closed_frames) / len(valid_frames)

class eye_features:
    def __init__(self, frame_width, frame_height, mar_threshold=0.2, perclos_threshold=0.2):
        self.mar_threshold = mar_threshold
        self.perclos_threshold = perclos_threshold
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Use RunningStat for efficient statistical calculations
        self.ear_stat = RunningStat(30)
        self.mar_stat = RunningStat(40)  # Added MAR statistics tracker
        self.left_pupil_stat = RunningStat(30)
        self.right_pupil_stat = RunningStat(30)
        
        # Specialized detectors
        self.blink_detector = BlinkDetector(20)
        self.perclos_calculator = PerclosCalculator(30, perclos_threshold)
        
        # Pre-calculate landmark indices for faster lookup
        self.left_eye_indices = [33, 133, 160, 158, 144, 153]
        self.right_eye_indices = [362, 263, 385, 387, 380, 373]
        self.left_pupil_indices = [469, 470, 471, 472]
        self.right_pupil_indices = [474, 475, 476, 477]
        self.mouth_indices = [67, 73, 11, 303, 61, 403, 16, 180]

    @staticmethod
    def euclidean_distance_2d(point1, point2):
        """Optimized Euclidean distance calculation."""
        x1, y1, _ = point1
        x2, y2, _ = point2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @staticmethod
    def euclidean_distance_3d(point1, point2):
        """Optimized Euclidean distance calculation."""
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def get_landmarks(self, face_results):
        """Get all landmarks in one pass for efficiency."""
        if not face_results.multi_face_landmarks:
            return None, None, None, None, None
            
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Extract landmarks more efficiently
        left_eye = [(face_landmarks.landmark[idx].x * self.frame_width,
                     face_landmarks.landmark[idx].y * self.frame_height,
                     face_landmarks.landmark[idx].z)
                    for idx in self.left_eye_indices]
                    
        right_eye = [(face_landmarks.landmark[idx].x * self.frame_width,
                      face_landmarks.landmark[idx].y * self.frame_height,
                      face_landmarks.landmark[idx].z)
                     for idx in self.right_eye_indices]
                     
        left_pupil = [(face_landmarks.landmark[idx].x * self.frame_width,
                       face_landmarks.landmark[idx].y * self.frame_height,
                       face_landmarks.landmark[idx].z)
                      for idx in self.left_pupil_indices]
                      
        right_pupil = [(face_landmarks.landmark[idx].x * self.frame_width,
                        face_landmarks.landmark[idx].y * self.frame_height,
                        face_landmarks.landmark[idx].z)
                       for idx in self.right_pupil_indices]
                       
        mouth = [(face_landmarks.landmark[idx].x * self.frame_width,
                 face_landmarks.landmark[idx].y * self.frame_height,
                 face_landmarks.landmark[idx].z)
                for idx in self.mouth_indices]
                
        return left_eye, right_eye, left_pupil, right_pupil, mouth

    def no_visible_eyes(self, face_results):
        """Check if eyes are visible."""
        if not face_results.multi_face_landmarks:
            return True
        
        face_landmarks = face_results.multi_face_landmarks[0]
        left_eye_visible = face_landmarks.landmark[33].visibility > 0.5
        right_eye_visible = face_landmarks.landmark[362].visibility > 0.5
        
        return not (left_eye_visible and right_eye_visible)

    def EyeAspectRatio2D(self, eye):
        """Calculate Eye Aspect Ratio (2D)."""
        if not eye or len(eye) < 6:
            return 0
            
        # Direct index access for speed
        p1, p4 = eye[0], eye[1]
        p2, p3 = eye[2], eye[3]
        p6, p5 = eye[4], eye[5]

        dist_p2_p6 = self.euclidean_distance_2d(p2, p6)
        dist_p3_p5 = self.euclidean_distance_2d(p3, p5)
        dist_p1_p4 = self.euclidean_distance_2d(p1, p4)

        if dist_p1_p4 < 1e-5:
            return 0
            
        return (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)

    def EyeAspectRatio3D(self, eye):
        """Calculate Eye Aspect Ratio (3D)."""
        if not eye or len(eye) < 6:
            return 0
            
        # Direct index access for speed
        p1, p4 = eye[0], eye[1]
        p2, p3 = eye[2], eye[3]
        p6, p5 = eye[4], eye[5]

        dist_p2_p6 = self.euclidean_distance_3d(p2, p6)
        dist_p3_p5 = self.euclidean_distance_3d(p3, p5)
        dist_p1_p4 = self.euclidean_distance_3d(p1, p4)

        if dist_p1_p4 < 1e-5:
            return 0
            
        return (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)

    def center_point(self, points):
        """Calculate center point efficiently."""
        if not points:
            return None
            
        x_sum = y_sum = 0
        for p in points:
            x_sum += p[0]
            y_sum += p[1]
            
        return (x_sum / len(points), y_sum / len(points))

    def eye_pupil_distance(self, eye, pupil):
        """Calculate distance between eye and pupil centers."""
        if not eye or not pupil:
            return None
            
        eye_center = self.center_point(eye)
        pupil_center = self.center_point(pupil)
        
        if not eye_center or not pupil_center:
            return None
            
        # Use 2D distance for efficiency
        return np.sqrt((eye_center[0] - pupil_center[0])**2 + 
                       (eye_center[1] - pupil_center[1])**2)

    def MouthAspectRatio(self, mouth):
        """Calculate Mouth Aspect Ratio."""
        if not mouth or len(mouth) < 8:
            return 0.0
            
        # Direct index access for speed
        p1, p5 = mouth[0], mouth[4]
        p2, p8 = mouth[1], mouth[7]
        p3, p7 = mouth[2], mouth[6]
        p4, p6 = mouth[3], mouth[5]

        horizontal_dist = self.euclidean_distance_2d(p1, p5)
        if horizontal_dist < 1e-5:
            return 0.0
            
        mar = (self.euclidean_distance_2d(p2, p8) + 
               self.euclidean_distance_2d(p3, p7) +
               self.euclidean_distance_2d(p4, p6)) / (2 * horizontal_dist)
               
        return mar

    def is_eye_closed(self, eye, threshold=0.2):
        """Check if eye is closed based on aspect ratio."""
        aspect_ratio = self.EyeAspectRatio2D(eye)
        return aspect_ratio < threshold

    def calculate_eye_features(self, face_results, frame_id):
        """Calculate all eye features with optimized statistics."""
        # Get all landmarks in one efficient pass
        left_eye, right_eye, left_pupil, right_pupil, mouth = self.get_landmarks(face_results)
        
        # Initialize with default values
        feature_data = {
            "frame_id": frame_id,
            "left_eye_aspect_ratio": 0,
            "right_eye_aspect_ratio": 0,
            "left_eye_aspect_ratio_3d": 0,
            "right_eye_aspect_ratio_3d": 0,
            "left_eye_pupil_distance": 0,
            "right_eye_pupil_distance": 0,
            "no_visible_eyes": True,
            "left_eye_closed": True,
            "right_eye_closed": True,
            "mouth_aspect_ratio": 0,
            "mouth_aspect_ratio_mean": 0,
            "mouth_aspect_ratio_variance": 0,
            "eye_closure_during_yawn": False,
            "left_eye_pupil_movement": 0,
            "left_eye_pupil_variance": 0,
            "right_eye_pupil_movement": 0,
            "right_eye_pupil_variance": 0,
            "num_blinks": 0,
            "ear_mean": 0,
            "ear_variance": 0,
            "perclos": 0,
        }
        
        # Return early if no face detected
        if left_eye is None:
            return feature_data
            
        # Calculate basic metrics
        no_visible_eyes = self.no_visible_eyes(face_results)
        left_ear = self.EyeAspectRatio2D(left_eye)
        right_ear = self.EyeAspectRatio2D(right_eye)
        left_ear_3d = self.EyeAspectRatio3D(left_eye)
        right_ear_3d = self.EyeAspectRatio3D(right_eye)
        left_pupil_dist = self.eye_pupil_distance(left_eye, left_pupil)
        right_pupil_dist = self.eye_pupil_distance(right_eye, right_pupil)
        left_eye_closed = self.is_eye_closed(left_eye)
        right_eye_closed = self.is_eye_closed(right_eye)
        mouth_aspect_ratio = self.MouthAspectRatio(mouth)
        
        # Update running statistics efficiently
        self.ear_stat.update((left_ear + right_ear) / 2)
        self.mar_stat.update(mouth_aspect_ratio)  # Update MAR statistics
        self.left_pupil_stat.update(left_pupil_dist)
        self.right_pupil_stat.update(right_pupil_dist)
        
        # Update specialized trackers
        self.blink_detector.update(left_eye_closed, right_eye_closed)
        self.perclos_calculator.update(left_ear, right_ear, no_visible_eyes)
        
        # Calculate complex metrics
        eye_closure_during_yawn = (mouth_aspect_ratio > self.mar_threshold) and (left_eye_closed and right_eye_closed)
        
        # Populate feature data
        feature_data.update({
            "left_eye_aspect_ratio": left_ear,
            "right_eye_aspect_ratio": right_ear,
            "left_eye_aspect_ratio_3d": left_ear_3d,
            "right_eye_aspect_ratio_3d": right_ear_3d,
            "left_eye_pupil_distance": left_pupil_dist,
            "right_eye_pupil_distance": right_pupil_dist,
            "no_visible_eyes": no_visible_eyes,
            "left_eye_closed": left_eye_closed,
            "right_eye_closed": right_eye_closed,
            "mouth_aspect_ratio": mouth_aspect_ratio,
            "mouth_aspect_ratio_mean": self.mar_stat.mean,
            "mouth_aspect_ratio_variance": self.mar_stat.variance,
            "eye_closure_during_yawn": eye_closure_during_yawn,
            "left_eye_pupil_movement": self.left_pupil_stat.mean,
            "left_eye_pupil_variance": self.left_pupil_stat.variance,
            "right_eye_pupil_movement": self.right_pupil_stat.mean,
            "right_eye_pupil_variance": self.right_pupil_stat.variance,
            "num_blinks": self.blink_detector.count_blinks(),
            "ear_mean": self.ear_stat.mean,
            "ear_variance": self.ear_stat.variance,
            "perclos": self.perclos_calculator.calculate(),
        })
        
        return feature_data