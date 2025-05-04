import cv2
import mediapipe as mp
import time
from memory_profiler import memory_usage
import psutil
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture("/home/harsh/Downloads/sem2/edgeai/video.mp4")

# Set the starting point to 2 minutes and 17 seconds (137 seconds)
start_time = 137  # in seconds
cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

end_time = start_time + 60  # in seconds

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "/home/harsh/Downloads/sem2/edgeai/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

# Variables for FPS calculation
prev_frame_time = 0
curr_frame_time = 0

mem_usage = []
last_mem_check = time.time()
mem_check_interval = 1.0  # Check memory every 1 second

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video, looping back to start")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    curr_frame_time = time.time()
    fps = 1/(curr_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = curr_frame_time

    if time.time() - last_mem_check >= mem_check_interval:
        current_mem = memory_usage(-1, interval=0.01, timeout=0.01)[0]
        mem_usage.append(current_mem)
        last_mem_check = time.time()
        
        cpu_percent = psutil.cpu_percent()
        
        print(f"Memory usage: {current_mem:.2f} MiB | CPU: {cpu_percent:.1f}%")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_frame)

    face_results = face_mesh.process(rgb_frame)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if mem_usage:
        cv2.putText(frame, f"Memory: {mem_usage[-1]:.2f} MiB", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)


    if cv2.waitKey(10) & 0xFF == 27: 
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
