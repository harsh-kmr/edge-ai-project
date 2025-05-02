import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque

import numpy as np
import cv2
from collections import deque
class head_features():
    def __init__(self, frame_width=640, frame_height=480,
                 baseline_yaw=5, baseline_pitch=-110, baseline_roll=10,
                 yaw_threshold=15, pitch_threshold=20, roll_threshold=20,
                 min_away_duration=10, buffer_size=30, event_window=300):
        """
        Initialize head features tracker with optimized thresholds for interior rear-view mirror camera.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
            baseline_yaw: Baseline yaw angle for normal driving (typically 5 degrees right)
            baseline_pitch: Baseline pitch angle for normal driving (typically -110 degrees)
            baseline_roll: Baseline roll angle for normal driving (typically 10 degrees)
            yaw_threshold: Deviation threshold from baseline yaw to detect distraction
            pitch_threshold: Deviation threshold from baseline pitch to detect distraction
            roll_threshold: Deviation threshold from baseline roll to detect distraction
            min_away_duration: Minimum frames to consider as distraction event
            buffer_size: Buffer size for variance calculation
            event_window: Window size for counting distraction events
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_id = 0
        
        # Baseline angles for normal driving position
        self.baseline_yaw = baseline_yaw
        self.baseline_pitch = baseline_pitch
        self.baseline_roll = baseline_roll
        
        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),              # Nose tip
            (0.0, -330.0, -65.0),         # Chin
            (-225.0, 170.0, -135.0),      # Left eye left corner
            (225.0, 170.0, -135.0),       # Right eye right corner
            (-150.0, -150.0, -125.0),     # Left mouth corner
            (150.0, -150.0, -125.0)       # Right mouth corner
        ])

        # Camera intrinsic parameters
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))

        # Buffers for angle tracking
        self.yaw_buffer = deque(maxlen=buffer_size)
        self.pitch_buffer = deque(maxlen=buffer_size)
        self.roll_buffer = deque(maxlen=buffer_size)

        # Thresholds and counters
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.current_away_duration = 0
        self.head_away_event_count = 0
        self.min_away_duration = min_away_duration
        self.event_window = event_window
        self.head_away_events_window = deque(maxlen=event_window)
        
        # Additional metrics
        self.distraction_confidence = 0.0
        self.distraction_pattern_detected = False

    def get_landmarks(self, face_landmarks):
        """Extract facial landmarks for head pose estimation."""
        idxs = [1, 199, 130, 359, 61, 291]  # Key facial points
        points = []
        for idx in idxs:
            lm = face_landmarks.landmark[idx]
            points.append((lm.x * self.frame_width, lm.y * self.frame_height))
        return np.array(points, dtype='double')

    def estimate_head_pose(self, face_landmarks):
        """Estimate head pose angles from facial landmarks."""
        if face_landmarks is None:
            return None, None, None

        try:
            image_points = self.get_landmarks(face_landmarks)
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeffs)

            if not success:
                return None, None, None

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            yaw, pitch, roll = self.rotation_matrix_to_euler_angles(rotation_matrix)

            # Convert to degrees
            yaw, pitch, roll = [angle * 180.0 / np.pi for angle in (yaw, pitch, roll)]

            # Store in buffer for variance calculation
            self.yaw_buffer.append(yaw)
            self.pitch_buffer.append(pitch)
            self.roll_buffer.append(roll)

            return yaw, pitch, roll
            
        except Exception as e:
            print(f"Error in head pose estimation: {e}")
            return None, None, None

    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """Convert rotation matrix to Euler angles."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return y, x, z  # yaw, pitch, roll

    def calculate_variances(self):
        """Calculate the variance of head pose angles over the buffer window."""
        if len(self.yaw_buffer) < 3:
            return 0.0, 0.0, 0.0
        return np.var(self.yaw_buffer), np.var(self.pitch_buffer), np.var(self.roll_buffer)

    def is_head_away(self, yaw, pitch, roll):
        """
        Determine if head position indicates distraction based on deviation from baseline.
        
        Returns:
            bool: True if head position indicates distraction
            float: Confidence score for distraction (0.0-1.0)
        """
        if yaw is None or pitch is None or roll is None:
            return True, 1.0
        
        # Calculate deviation from baseline normal driving position
        yaw_deviation = abs(yaw - self.baseline_yaw)
        pitch_deviation = abs(pitch - self.baseline_pitch)
        roll_deviation = abs(roll - self.baseline_roll)
        
        # Calculate distraction confidence score (0.0-1.0)
        yaw_factor = min(1.0, yaw_deviation / (self.yaw_threshold * 2))
        pitch_factor = min(1.0, pitch_deviation / (self.pitch_threshold * 2))
        roll_factor = min(1.0, roll_deviation / (self.roll_threshold * 2))
        
        # Combined confidence score (weighted average)
        confidence = 0.5 * yaw_factor + 0.3 * pitch_factor + 0.2 * roll_factor
        
        # Determine if head is away based on thresholds
        is_away = (yaw_deviation > self.yaw_threshold or 
                  pitch_deviation > self.pitch_threshold or 
                  roll_deviation > self.roll_threshold)
        
        return is_away, confidence

    def update_head_away_status(self, frame_id, yaw, pitch, roll):
        """Update head away status and track distraction events."""
        self.frame_id = frame_id
        
        is_away, confidence = self.is_head_away(yaw, pitch, roll)
        self.distraction_confidence = confidence
        
        if is_away:
            self.current_away_duration += 1
        else:
            # Record event if duration threshold met
            if self.current_away_duration >= self.min_away_duration:
                self.head_away_event_count += 1
                self.head_away_events_window.append(frame_id)
            self.current_away_duration = 0

        

        return self.current_away_duration
    
    def count_head_away_events(self, frame_id):
        """Count distraction events within the time window."""
        # Count events within the time window relative to the current frame
        return sum(1 for event_frame in self.head_away_events_window 
                  if frame_id - event_frame <= self.event_window)


    
    def calculate_head_features(self, face_results, frame, frame_id):
        """Calculate all head features and return metrics."""
        yaw, pitch, roll = None, None, None
        nose_point = (self.frame_width // 2, self.frame_height // 2)

        # Extract face landmarks if available
        if hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            nose = face_landmarks.landmark[1]  # Nose tip landmark
            nose_point = (int(nose.x * self.frame_width), int(nose.y * self.frame_height))
            yaw, pitch, roll = self.estimate_head_pose(face_landmarks)

        # Update distraction metrics
        head_away_duration = self.update_head_away_status(frame_id, yaw, pitch, roll)
        yaw_var, pitch_var, roll_var = self.calculate_variances()
        head_away_count = self.count_head_away_events(frame_id)

        # print(f"Frame {frame_id}: Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}, ", 
        #       f"yaw_deviation: {abs(yaw - self.baseline_yaw)}, ",
        #       f"pitch_deviation: {abs(pitch - self.baseline_pitch)}, ",
        #       f"roll_deviation: {abs(roll - self.baseline_roll)}, ",
        #       f"baseline yaw: {self.baseline_yaw}, ",
        #       f"baseline pitch: {self.baseline_pitch}, ",
        #       f"baseline roll: {self.baseline_roll}, ",)

        # Return comprehensive feature dictionary
        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "yaw_variance": yaw_var,
            "pitch_variance": pitch_var,
            "roll_variance": roll_var,
            "head_away_duration": head_away_duration,
            "head_away_event_count": head_away_count,
            "distraction_confidence": self.distraction_confidence,
            # Add baseline deviations if angles are available
            "yaw_deviation": abs(yaw - self.baseline_yaw) if yaw is not None else None,
            "pitch_deviation": abs(pitch - self.baseline_pitch) if pitch is not None else None,
            "roll_deviation": abs(roll - self.baseline_roll) if roll is not None else None
        }

def process_video_head(video_path, 
                       baseline_yaw=5, baseline_pitch=-110, baseline_roll=10,
                       yaw_threshold=15, pitch_threshold=20, roll_threshold=20,
                       min_away_duration=10, buffer_size=30, event_window=300):
    df = pd.DataFrame()
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize head_features with all parameters
    head_feature = head_features(
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
    
    frame_id = 0
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        face_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        features = head_feature.calculate_head_features(face_results, frame, frame_id=frame_id)

        df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
        frame_id += 1
        
    cap.release()
    cv2.destroyAllWindows()
    return df
    
if __name__ == "__main__":
    df = process_video_head(
        video_path="/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/normal_5.mp4",
        baseline_yaw=5,
        baseline_pitch=-110,
        baseline_roll=10,
        yaw_threshold=30,
        pitch_threshold=30,
        roll_threshold=20,
        min_away_duration=15,
        buffer_size=30,
        event_window=300
    )

    print(df.head())