import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
import time
import threading
import queue

class head_features:
    def __init__(self, frame_width=640, frame_height=480,
                 baseline_yaw=5, baseline_pitch=-110, baseline_roll=10,
                 yaw_threshold=15, pitch_threshold=20, roll_threshold=20,
                 min_away_duration=10, buffer_size=30, event_window=300):
        """
        Initialize head features tracker optimized for Raspberry Pi.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
            baseline_yaw: Baseline yaw angle for normal driving
            baseline_pitch: Baseline pitch angle for normal driving
            baseline_roll: Baseline roll angle for normal driving
            yaw_threshold: Deviation threshold from baseline yaw
            pitch_threshold: Deviation threshold from baseline pitch
            roll_threshold: Deviation threshold from baseline roll
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
        
        # 3D model points for head pose estimation - simplified for performance
        self.model_points = np.array([
            (0.0, 0.0, 0.0),              # Nose tip
            (0.0, -330.0, -65.0),         # Chin
            (-225.0, 170.0, -135.0),      # Left eye left corner
            (225.0, 170.0, -135.0),       # Right eye right corner
            (-150.0, -150.0, -125.0),     # Left mouth corner
            (150.0, -150.0, -125.0)       # Right mouth corner
        ], dtype=np.float32)  # Use float32 for better performance on Raspberry Pi

        # Camera intrinsic parameters
        center = (frame_width / 2, frame_height / 2)
        focal_length = frame_width  # Approximation
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)  # Use float32

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # Use RunningStat for efficient variance calculation
        self.yaw_stat = RunningStat(buffer_size)
        self.pitch_stat = RunningStat(buffer_size)
        self.roll_stat = RunningStat(buffer_size)

        # Thresholds and counters
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.current_away_duration = 0
        self.head_away_event_count = 0
        self.min_away_duration = min_away_duration
        self.event_window = event_window
        
        # Use a simple list instead of deque for event tracking
        self.head_away_events = []
        self.max_events = event_window
        
        # Additional metrics
        self.distraction_confidence = 0.0
        
        # Reuse landmark indices for improved performance
        self.landmark_indices = [1, 199, 130, 359, 61, 291]  # Key facial points

    def get_landmarks(self, face_landmarks):
        """Extract facial landmarks for head pose estimation."""
        points = []
        for idx in self.landmark_indices:
            lm = face_landmarks.landmark[idx]
            points.append((lm.x * self.frame_width, lm.y * self.frame_height))
        return np.array(points, dtype=np.float32)  # Use float32

    def estimate_head_pose(self, face_landmarks):
        """Estimate head pose angles from facial landmarks."""
        if face_landmarks is None:
            return None, None, None

        try:
            image_points = self.get_landmarks(face_landmarks)
            
            # Use more efficient SOLVEPNP_IPPE algorithm for Raspberry Pi
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE)

            if not success:
                return None, None, None

            # More efficient conversion of rotation vector to matrix
            rotation_matrix = np.zeros((3, 3), dtype=np.float32)
            cv2.Rodrigues(rotation_vector, rotation_matrix)
            
            yaw, pitch, roll = self.rotation_matrix_to_euler_angles(rotation_matrix)

            # Convert to degrees
            yaw_deg = yaw * 180.0 / np.pi
            pitch_deg = pitch * 180.0 / np.pi
            roll_deg = roll * 180.0 / np.pi

            # Update running statistics
            self.yaw_stat.update(yaw_deg)
            self.pitch_stat.update(pitch_deg)
            self.roll_stat.update(roll_deg)

            return yaw_deg, pitch_deg, roll_deg
            
        except Exception as e:
            # Avoid printing errors in production code - slows down processing
            return None, None, None

    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """Convert rotation matrix to Euler angles - optimized version."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy > 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
            
        return y, x, z  # yaw, pitch, roll

    def is_head_away(self, yaw, pitch, roll):
        """Determine if head position indicates distraction."""
        if yaw is None or pitch is None or roll is None:
            return True, 1.0
        
        # Calculate deviation from baseline
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
                self.head_away_events.append(frame_id)
                
                # Keep only the most recent events within window
                if len(self.head_away_events) > self.max_events:
                    self.head_away_events.pop(0)
                    
            self.current_away_duration = 0

        return self.current_away_duration
    
    def count_head_away_events(self, frame_id):
        """Count distraction events within the time window."""
        cutoff_frame = frame_id - self.event_window
        
        # Filter and count events within window
        count = sum(1 for event_frame in self.head_away_events if event_frame > cutoff_frame)
        return count
    
    def calculate_head_features(self, face_results, frame_id):
        """Calculate all head features and return metrics - optimized version."""
        yaw, pitch, roll = None, None, None

        # Extract face landmarks if available
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            yaw, pitch, roll = self.estimate_head_pose(face_landmarks)

        # Update distraction metrics
        head_away_duration = self.update_head_away_status(frame_id, yaw, pitch, roll)
        head_away_count = self.count_head_away_events(frame_id)

        # Return only essential metrics
        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "yaw_variance": self.yaw_stat.variance if self.yaw_stat.is_full else 0.0,
            "pitch_variance": self.pitch_stat.variance if self.pitch_stat.is_full else 0.0,
            "roll_variance": self.roll_stat.variance if self.roll_stat.is_full else 0.0,
            "head_away_duration": head_away_duration,
            "head_away_event_count": head_away_count,
            "distraction_confidence": self.distraction_confidence,
            "yaw_deviation": abs(yaw - self.baseline_yaw) if yaw is not None else None,
            "pitch_deviation": abs(pitch - self.baseline_pitch) if pitch is not None else None,
            "roll_deviation": abs(roll - self.baseline_roll) if roll is not None else None
        }


class VideoProcessor:
    """Thread-based video processor for better performance on Raspberry Pi."""
    def __init__(self, video_path, output_path=None, 
                 baseline_yaw=5, baseline_pitch=-110, baseline_roll=10,
                 yaw_threshold=15, pitch_threshold=20, roll_threshold=20,
                 min_away_duration=10, buffer_size=30, event_window=300,
                 resize_factor=0.5, process_every_n_frames=2,
                 display=False):
        """
        Initialize video processor with threading support.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video (optional)
            baseline_*: Baseline angles for normal driving position
            *_threshold: Deviation thresholds for angles
            min_away_duration: Minimum frames to consider distraction
            buffer_size: Buffer size for variance calculation
            event_window: Window size for counting distraction events
            resize_factor: Factor to resize frames (smaller = faster)
            process_every_n_frames: Process every N frames (skip frames)
            display: Whether to display processed frames
        """
        self.video_path = video_path
        self.output_path = output_path
        self.resize_factor = resize_factor
        self.process_every_n_frames = process_every_n_frames
        self.display = display
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize head features tracker
        self.head_feature = HeadFeatures(
            frame_width=self.frame_width, 
            frame_height=self.frame_height,
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
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        # Use lighter model configuration for Raspberry Pi
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Disable refinement for speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize result storage
        self.results = []
        
        # Setup for video writer if output path is provided
        self.video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps / process_every_n_frames,
                (self.frame_width, self.frame_height)
            )
            
        # Threading components
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer max 30 frames
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        
    def frame_producer(self):
        """Thread function to read frames from video."""
        frame_id = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process only every N frames
            if frame_id % self.process_every_n_frames == 0:
                # Resize for better performance
                if self.resize_factor != 1.0:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                try:
                    # Add to queue with timeout to avoid blocking indefinitely
                    self.frame_queue.put((frame_id, frame), timeout=1)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                    
            frame_id += 1
            
        # Signal end of frames
        self.frame_queue.put((None, None))
            
    def frame_processor(self):
        """Thread function to process frames."""
        while not self.stop_event.is_set():
            try:
                # Get frame from queue with timeout
                frame_id, frame = self.frame_queue.get(timeout=1)
                if frame_id is None:  # End signal
                    break
                    
                # Convert to RGB only once
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                face_results = self.face_mesh.process(rgb_frame)
                
                # Calculate head features
                features = self.head_feature.calculate_head_features(face_results, frame_id)
                features['frame_id'] = frame_id
                
                # Add to results queue
                self.result_queue.put((frame_id, frame, features))
                
                # Mark task as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                if frame_id is not None:
                    self.frame_queue.task_done()
                    
        # Signal end of processing
        self.result_queue.put((None, None, None))