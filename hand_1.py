import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pandas as pd
import threading
import queue

from tools import RunningStat
import time

class hand_features:
    def __init__(self, frame_width=640, frame_height=480, 
                 proximity_threshold=0.2, 
                 alpha=1.8, beta=1.5,  
                 default_wheel_box=[0.25, 0.92, 0.4, 0.6],  
                 window_size=150): 
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.proximity_threshold = proximity_threshold
        self.default_wheel_box = default_wheel_box  # [x_min, x_max, y_min, y_max] normalized
        self.alpha = alpha  # Steering wheel width scaling factor
        self.beta = beta    # Vertical position factor
        self.window_size = window_size
        
        # Use RunningStat for time-based features
        self.left_hand_off_wheel_stat = RunningStat(window_size)
        self.right_hand_off_wheel_stat = RunningStat(window_size)
        self.left_hand_distance_stat = RunningStat(window_size)
        self.right_hand_distance_stat = RunningStat(window_size)
        
        # Current calculated wheel box
        self.current_wheel_box = default_wheel_box.copy()
        
        # For thread-safe operation
        self.lock = threading.Lock()
    
    def calculate_wheel_box(self, pose_landmarks):
        """Calculate dynamic steering wheel box based on body pose landmarks"""
        wheel_box = self.default_wheel_box.copy()
        
        # Only calculate if we have valid pose landmarks
        if pose_landmarks and pose_landmarks.pose_landmarks:
            landmarks = pose_landmarks.pose_landmarks.landmark
            
            # Check if we have the required shoulder and hip points
            if (landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5 and 
                landmarks[23].visibility > 0.5 and landmarks[24].visibility > 0.5):
                
                # Extract coordinates (no need to recalculate multiple times)
                x11, y11 = landmarks[11].x, landmarks[11].y  # Left shoulder
                x12, y12 = landmarks[12].x, landmarks[12].y  # Right shoulder
                x23, y23 = landmarks[23].x, landmarks[23].y  # Left hip
                x24, y24 = landmarks[24].x, landmarks[24].y  # Right hip
                
                # Compute wheel box parameters using alpha and beta
                shoulder_width = abs(x12 - x11)
                W = self.alpha * shoulder_width
                H = W * 0.6
                
                # Calculate center points
                Cx = (x11 + x12) * 0.5
                shoulder_y = (y11 + y12) * 0.5
                hip_y = (y23 + y24) * 0.5
                Cy = shoulder_y + self.beta * (hip_y - shoulder_y)
                
                # Calculate box in normalized coordinates
                wheel_box = [
                    max(0.0, Cx - W/2),          # xmin
                    min(1.0, Cx + W/2),          # xmax
                    max(0.0, Cy - 3*H/4),        # ymin
                    min(1.0, Cy + H/4)           # ymax
                ]
        
        # Store and return
        self.current_wheel_box = wheel_box
        return wheel_box
    
    def extract_hand_position(self, pose_landmarks):
        # Initialize with default values
        hand_pos = {
            "left_wrist_x": 0, "left_wrist_y": 0, "left_wrist_z": 0,
            "left_palm_x": 0, "left_palm_y": 0, "left_palm_z": 0,
            "right_wrist_x": 0, "right_wrist_y": 0, "right_wrist_z": 0,
            "right_palm_x": 0, "right_palm_y": 0, "right_palm_z": 0
        }
        
        if not pose_landmarks or not pose_landmarks.pose_landmarks:
            return hand_pos
        
        landmarks = pose_landmarks.pose_landmarks.landmark
        
        # Left hand: landmarks 15 (wrist), 17 and 19 (palm)
        if landmarks[15].visibility > 0.5:  # Check if left wrist is visible
            hand_pos["left_wrist_x"] = landmarks[15].x * self.frame_width
            hand_pos["left_wrist_y"] = landmarks[15].y * self.frame_height
            hand_pos["left_wrist_z"] = landmarks[15].z
            
            # Calculate palm center as average of landmarks 17 and 19
            if landmarks[17].visibility > 0.5 and landmarks[19].visibility > 0.5:
                palm_x = (landmarks[17].x + landmarks[19].x) * 0.5
                palm_y = (landmarks[17].y + landmarks[19].y) * 0.5
                palm_z = (landmarks[17].z + landmarks[19].z) * 0.5
                
                hand_pos["left_palm_x"] = palm_x * self.frame_width
                hand_pos["left_palm_y"] = palm_y * self.frame_height
                hand_pos["left_palm_z"] = palm_z
        
        # Right hand: landmarks 16 (wrist), 18 and 20 (palm)
        if landmarks[16].visibility > 0.5:  # Check if right wrist is visible
            hand_pos["right_wrist_x"] = landmarks[16].x * self.frame_width
            hand_pos["right_wrist_y"] = landmarks[16].y * self.frame_height
            hand_pos["right_wrist_z"] = landmarks[16].z
            
            # Calculate palm center as average of landmarks 18 and 20
            if landmarks[18].visibility > 0.5 and landmarks[20].visibility > 0.5:
                palm_x = (landmarks[18].x + landmarks[20].x) * 0.5
                palm_y = (landmarks[18].y + landmarks[20].y) * 0.5
                palm_z = (landmarks[18].z + landmarks[20].z) * 0.5
                
                hand_pos["right_palm_x"] = palm_x * self.frame_width
                hand_pos["right_palm_y"] = palm_y * self.frame_height
                hand_pos["right_palm_z"] = palm_z
                
        return hand_pos

    def compute_hand_eye_depth(self, pose_landmarks, face_landmarks):
        depths = {"left_hand_eye_depth": 0.0, "right_hand_eye_depth": 0.0}
        
        if not pose_landmarks or not pose_landmarks.pose_landmarks or not face_landmarks or not face_landmarks.multi_face_landmarks:
            return depths
        
        # Calculate eye center
        try:
            left_eye = face_landmarks.multi_face_landmarks[0].landmark[468]
            right_eye = face_landmarks.multi_face_landmarks[0].landmark[473]
            
            # Precompute eye center coordinates
            eye_x = (left_eye.x + right_eye.x) * 0.5 * self.frame_width
            eye_y = (left_eye.y + right_eye.y) * 0.5 * self.frame_height
            eye_z = (left_eye.z + right_eye.z) * 0.5
            
            landmarks = pose_landmarks.pose_landmarks.landmark
            
            # Calculate depth for left hand (wrist)
            if landmarks[15].visibility > 0.5:  # Left wrist
                left_x = landmarks[15].x * self.frame_width
                left_y = landmarks[15].y * self.frame_height
                left_z = landmarks[15].z
                
                # Use faster Euclidean distance calculation
                dx = left_x - eye_x
                dy = left_y - eye_y
                dz = left_z - eye_z
                depths["left_hand_eye_depth"] = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Calculate depth for right hand (wrist)
            if landmarks[16].visibility > 0.5:  # Right wrist
                right_x = landmarks[16].x * self.frame_width
                right_y = landmarks[16].y * self.frame_height
                right_z = landmarks[16].z
                
                dx = right_x - eye_x
                dy = right_y - eye_y
                dz = right_z - eye_z
                depths["right_hand_eye_depth"] = np.sqrt(dx*dx + dy*dy + dz*dz)
        except (IndexError, AttributeError):
            # Handle potential errors with the face landmarks
            pass
            
        return depths

    def calculate_angle_3d(self, p1, p2, p3):
        """Calculate the angle between three 3D points, with p2 as the vertex"""
        # Create vectors (p1-p2 and p3-p2)
        v1 = [p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]]
        v2 = [p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]]
        
        # Calculate dot product
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        
        # Calculate magnitudes
        mag1 = np.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
        mag2 = np.sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2])
        
        # Calculate angle
        if mag1 * mag2 < 1e-10:  # Avoid division by zero
            return 0.0
        
        cos_angle = dot / (mag1 * mag2)
        # Clamp to avoid numerical issues
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return np.arccos(cos_angle) * 180 / np.pi

    def extract_arm_angles(self, pose_landmarks):
        """Extract left and right elbow angles (shoulder-elbow-wrist)"""
        angles = {"left_elbow_angle": 0.0, "right_elbow_angle": 0.0}
        
        if not pose_landmarks or not pose_landmarks.pose_landmarks:
            return angles
        
        landmarks = pose_landmarks.pose_landmarks.landmark
        
        # Extract 3D points for right arm (shoulder 12, elbow 14, wrist 16)
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
        
        # Extract 3D points for left arm (shoulder 11, elbow 13, wrist 15)
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
        # Calculate the current wheel box
        current_wheel_box = self.calculate_wheel_box(pose_landmarks)
        
        # Default values for the result
        left_off_wheel = 1  # Default to off wheel
        right_off_wheel = 1  # Default to off wheel
        
        if pose_landmarks and pose_landmarks.pose_landmarks:
            # Convert normalized wheel box to pixel coordinates for faster comparison
            x_min = current_wheel_box[0] * self.frame_width
            x_max = current_wheel_box[1] * self.frame_width
            y_min = current_wheel_box[2] * self.frame_height
            y_max = current_wheel_box[3] * self.frame_height
            
            landmarks = pose_landmarks.pose_landmarks.landmark
            
            # Check left wrist (landmark 15)
            if landmarks[15].visibility > 0.5:
                x, y = landmarks[15].x * self.frame_width, landmarks[15].y * self.frame_height
                
                # Check if hand is within the steering wheel area
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    left_off_wheel = 0  # Hand is on wheel
            
            # Check right wrist (landmark 16)
            if landmarks[16].visibility > 0.5:
                x, y = landmarks[16].x * self.frame_width, landmarks[16].y * self.frame_height
                
                # Check if hand is within the steering wheel area
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    right_off_wheel = 0  # Hand is on wheel
        
        # Update our running statistics
        self.left_hand_off_wheel_stat.update(left_off_wheel)
        self.right_hand_off_wheel_stat.update(right_off_wheel)
        
        # Return the current values and durations
        return {
            "left_hand_off_wheel": bool(left_off_wheel),
            "right_hand_off_wheel": bool(right_off_wheel),
            "left_hand_off_wheel_duration": self.left_hand_off_wheel_stat.sum,
            "right_hand_off_wheel_duration": self.right_hand_off_wheel_stat.sum
        }
    
    def check_hand_proximity_to_face(self, pose_landmarks, face_landmarks):
        # Initialize result dictionary
        result = {
            "left_hand_distance": 0,
            "right_hand_distance": 0,
            "left_hand_distance_var": 0,
            "right_hand_distance_var": 0
        }
        
        # If no face or pose detected, return default values
        if (not pose_landmarks or not pose_landmarks.pose_landmarks or 
            not face_landmarks or not face_landmarks.multi_face_landmarks):
            # Update stats with 0 (hand not detected)
            self.left_hand_distance_stat.update(0)
            self.right_hand_distance_stat.update(0)
            return result
        
        try:
            # Get nose position as reference point for face
            nose = face_landmarks.multi_face_landmarks[0].landmark[1]  # Nose tip landmark
            nose_x = nose.x * self.frame_width
            nose_y = nose.y * self.frame_height
            
            landmarks = pose_landmarks.pose_landmarks.landmark
            
            # Check left wrist (landmark 15)
            if landmarks[15].visibility > 0.5:
                wrist_x = landmarks[15].x * self.frame_width
                wrist_y = landmarks[15].y * self.frame_height
                
                # Calculate Euclidean distance between wrist and nose
                dx = wrist_x - nose_x
                dy = wrist_y - nose_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                result["left_hand_distance"] = distance
                self.left_hand_distance_stat.update(distance)
            else:
                # Left hand not detected
                self.left_hand_distance_stat.update(0)
                
            # Check right wrist (landmark 16)
            if landmarks[16].visibility > 0.5:
                wrist_x = landmarks[16].x * self.frame_width
                wrist_y = landmarks[16].y * self.frame_height
                
                # Calculate Euclidean distance between wrist and nose
                dx = wrist_x - nose_x
                dy = wrist_y - nose_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                result["right_hand_distance"] = distance
                self.right_hand_distance_stat.update(distance)
            else:
                # Right hand not detected
                self.right_hand_distance_stat.update(0)
            
            # Calculate variance using our RunningStat
            result["left_hand_distance_var"] = self.left_hand_distance_stat.variance
            result["right_hand_distance_var"] = self.right_hand_distance_stat.variance
            
        except (IndexError, AttributeError):
            # Handle potential face landmark errors
            pass
            
        return result

    def calculate_hand_features(self, pose_results, face_landmarks):
        """Calculate all hand features in one pass"""
        with self.lock:
            # Extract all features
            hand_pos = self.extract_hand_position(pose_results)
            hand_eye_depth = self.compute_hand_eye_depth(pose_results, face_landmarks)
            arm_angles = self.extract_arm_angles(pose_results)
            hand_off_wheel = self.check_hand_off_steering_wheel(pose_results)
            hand_proximity = self.check_hand_proximity_to_face(pose_results, face_landmarks)
            
            # Include wheel box information in the output
            wheel_box_info = {
                "wheel_box_x_min": self.current_wheel_box[0],
                "wheel_box_x_max": self.current_wheel_box[1],
                "wheel_box_y_min": self.current_wheel_box[2],
                "wheel_box_y_max": self.current_wheel_box[3]
            }
            
            # Combine all features into a single dictionary
            data = {
                **hand_pos,
                **hand_eye_depth,
                **arm_angles,
                **hand_off_wheel,
                **hand_proximity,
                **wheel_box_info
            }
            
            return data


class VideoProcessor:
    """Class to handle video processing with threading for Raspberry Pi"""
    def __init__(self, video_path=None, window_size=150):
        self.video_path = video_path
        self.window_size = window_size
        self.frame_queue = queue.Queue(maxsize=5)  # Small queue to prevent memory issues
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_complete = threading.Event()
        
        # Initialize MediaPipe with lite models for Raspberry Pi
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        
    def capture_frames(self):
        """Thread function to capture frames from video"""
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB once (more efficient)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # If queue is full, wait a bit
                if self.frame_queue.full():
                    time.sleep(0.01)
                    
                # Put frame data and metadata in queue
                self.frame_queue.put((rgb_frame, frame_width, frame_height))
        finally:
            cap.release()
            # Signal end of frames
            self.frame_queue.put(None)
    
    def process_frames(self):
        """Thread function to process frames"""
        # Initialize models with lightweight configurations for Raspberry Pi
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0  # Use the lightest model for Raspberry Pi
            ) as pose_mesh:
                frame_id = 0
                results_list = []
                
                while not self.stop_event.is_set():
                    # Get frame data from queue
                    queue_item = self.frame_queue.get()
                    
                    # Check for end signal
                    if queue_item is None:
                        self.frame_queue.task_done()
                        break
                    
                    rgb_frame, frame_width, frame_height = queue_item
                    
                    # Process frame with MediaPipe
                    face_results = face_mesh.process(rgb_frame)
                    pose_results = pose_mesh.process(rgb_frame)
                    
                    # Initialize features processor if not already done
                    if frame_id == 0:
                        self.features = OptimizedHandFeatures(
                            frame_width=frame_width,
                            frame_height=frame_height,
                            window_size=self.window_size
                        )
                    
                    # Calculate features
                    features = self.features.calculate_hand_features(pose_results, face_results)
                    features['frame_id'] = frame_id
                    
                    # Collect results
                    results_list.append(features)
                    
                    # Batch processing - only process every few frames to reduce load
                    if len(results_list) >= 100:
                        df_batch = pd.DataFrame(results_list)
                        self.result_queue.put(df_batch)
                        results_list = []
                    
                    frame_id += 1
                    self.frame_queue.task_done()
                
                # Process any remaining results
                if results_list:
                    df_batch = pd.DataFrame(results_list)
                    self.result_queue.put(df_batch)
                
                # Signal end of processing
                self.result_queue.put(None)
                self.processing_complete.set()
    
    def process_video(self):
        """Process the video and return a DataFrame with features"""
        # Start threads
        capture_thread = threading.Thread(target=self.capture_frames)
        process_thread = threading.Thread(target=self.process_frames)
        
        capture_thread.start()
        process_thread.start()
        
        # Collect results
        all_results = []
        
        try:
            while not self.processing_complete.is_set() or not self.result_queue.empty():
                result = self.result_queue.get(timeout=1.0)
                if result is None:
                    self.result_queue.task_done()
                    break
                
                all_results.append(result)
                self.result_queue.task_done()
        except queue.Empty:
            pass
        finally:
            # Ensure threads are stopped
            self.stop_event.set()
            
            # Wait for threads to finish
            capture_thread.join()
            process_thread.join()
        
        # Combine all results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()


def process_video_hand_optimized(video_path, proximity_threshold=0.2, 
                            alpha=1.8, beta=1.5, 
                            default_wheel_box=[0.25, 0.92, 0.4, 0.6], 
                            window_size=150):
    """Optimized video processing function for Raspberry Pi"""
    processor = VideoProcessor(video_path=video_path, window_size=window_size)
    df = processor.process_video()
    return df


if __name__ == "__main__":
    # Example usage
    start_time = time.time()
    df = process_video_hand_optimized(
        video_path="your_video_path.mp4",
        window_size=150
    )
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Processed {len(df)} frames")
    print(df.head())
    
    # Save results if needed
    # df.to_csv("hand_features.csv", index=False)
