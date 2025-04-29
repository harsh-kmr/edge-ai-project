import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque

class HeadPoseEstimator:
    def __init__(self, frame_width=640, frame_height=480,
                 yaw_threshold=30, pitch_threshold=20, min_away_duration=15, buffer_size=30, event_window=300):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))

        self.yaw_buffer = deque(maxlen=buffer_size)
        self.pitch_buffer = deque(maxlen=buffer_size)
        self.roll_buffer = deque(maxlen=buffer_size)

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.current_away_duration = 0
        self.head_away_event_count = 0
        self.min_away_duration = min_away_duration
        self.event_window = event_window
        self.head_away_events_window = deque(maxlen=event_window)

        self.frame_id = 0

    def get_landmarks(self, face_landmarks):
        idxs = [1, 199, 130, 359, 61, 291]
        points = []
        for idx in idxs:
            lm = face_landmarks.landmark[idx]
            points.append((lm.x * self.frame_width, lm.y * self.frame_height))
        return np.array(points, dtype='double')

    def estimate_head_pose(self, face_landmarks):
        if face_landmarks is None:
            return None, None, None

        image_points = self.get_landmarks(face_landmarks)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs)

        if not success:
            return None, None, None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        yaw, pitch, roll = self.rotation_matrix_to_euler_angles(rotation_matrix)

        yaw, pitch, roll = [angle * 180.0 / np.pi for angle in (yaw, pitch, roll)]

        self.yaw_buffer.append(yaw)
        self.pitch_buffer.append(pitch)
        self.roll_buffer.append(roll)

        return yaw, pitch, roll

    @staticmethod
    def rotation_matrix_to_euler_angles(R):
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
        return y, x, z

    def calculate_variances(self):
        if len(self.yaw_buffer) < 3:
            return 0.0, 0.0, 0.0
        return np.var(self.yaw_buffer), np.var(self.pitch_buffer), np.var(self.roll_buffer)

    def update_head_away_status(self, yaw, pitch):
        if yaw is None or pitch is None:
            is_away = True
        else:
            is_away = abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold

        if is_away:
            self.current_away_duration += 1
        else:
            if self.current_away_duration >= self.min_away_duration:
                self.head_away_event_count += 1
                self.head_away_events_window.append(self.frame_id)
            self.current_away_duration = 0

        return self.current_away_duration

    def count_head_away_events(self):
        current_frame = self.frame_id
        return sum(1 for event_frame in self.head_away_events_window if current_frame - event_frame <= self.event_window)

    def draw_axes(self, frame, yaw, pitch, roll, nose_point):
        length = 50
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

        x1 = int(nose_point[0] + length * (np.cos(yaw) * np.cos(roll)))
        y1 = int(nose_point[1] + length * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)))
        x2 = int(nose_point[0] + length * (-np.cos(yaw) * np.sin(roll)))
        y2 = int(nose_point[1] + length * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)))
        x3 = int(nose_point[0] + length * np.sin(yaw))
        y3 = int(nose_point[1] + length * (-np.cos(yaw) * np.sin(pitch)))

        cv2.line(frame, nose_point, (x1, y1), (0, 0, 255), 2)
        cv2.line(frame, nose_point, (x2, y2), (0, 255, 0), 2)
        cv2.line(frame, nose_point, (x3, y3), (255, 0, 0), 2)

    def process_frame(self, frame, face_mesh):
        self.frame_id += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        yaw, pitch, roll = None, None, None
        nose_point = (self.frame_width // 2, self.frame_height // 2)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            nose = face_landmarks.landmark[1]
            nose_point = (int(nose.x * self.frame_width), int(nose.y * self.frame_height))
            yaw, pitch, roll = self.estimate_head_pose(face_landmarks)

        head_away_duration = self.update_head_away_status(yaw, pitch)
        yaw_var, pitch_var, roll_var = self.calculate_variances()
        head_away_count = self.count_head_away_events()

        if yaw is not None and pitch is not None and roll is not None:
            self.draw_axes(frame, yaw, pitch, roll, nose_point)

        return {
            "frame_id": self.frame_id,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "yaw_variance": yaw_var,
            "pitch_variance": pitch_var,
            "roll_variance": roll_var,
            "head_away_duration": head_away_duration,
            "head_away_event_count": head_away_count
        }

if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    # Input Video
        
    cap = cv2.VideoCapture('/home/vikas/Project/code/MicrosoftTeams-video (2).mp4')
    head_pose_estimator = HeadPoseEstimator()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('head_pose_output.avi', fourcc, fps if fps > 0 else 30, (frame_width, frame_height))

    all_data = []

    
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = head_pose_estimator.process_frame(frame, face_mesh)
        all_data.append(data)

        
            # Draw features on frame
        if data['yaw'] is not None and data['pitch'] is not None and data['roll'] is not None:
            cv2.putText(frame, f"Frame: {data['frame_id']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Yaw: {data['yaw']:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, f"Pitch: {data['pitch']:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Roll: {data['roll']:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        if data['head_away_duration'] > 0:
            cv2.putText(frame, f"HEAD AWAY: {data['head_away_duration']} frames", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"Events: {data['head_away_event_count']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        out.write(frame)

        # cv2.imshow("Head Pose Estimation", frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(all_data)
    df.to_csv("head_pose_features.csv", index=False)
    print("Saved head pose features to head_pose_features.csv")
    print("Saved output video to head_pose_output.avi")

