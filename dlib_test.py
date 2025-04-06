import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_head_pose(face_landmarks):
    """Calculates head pose angles (pitch, yaw, roll) from face landmarks using frame dimensions."""
    if not face_landmarks:
        return None


    # Get relevant landmark coordinates
    image_points = np.array([
        (face_landmarks.landmark[4].x, face_landmarks.landmark[4].y),    # Nose tip
        (face_landmarks.landmark[152].x, face_landmarks.landmark[152].y),  # Chin
        (face_landmarks.landmark[234].x, face_landmarks.landmark[234].y),  # Left eye corner
        (face_landmarks.landmark[454].x, face_landmarks.landmark[454].y),  # Right eye corner
        (face_landmarks.landmark[127].x, face_landmarks.landmark[127].y),  # Left mouth corner
        (face_landmarks.landmark[356].x, face_landmarks.landmark[356].y)   # Right mouth corner
    ], dtype=np.double)

    # Use the global frame_width and frame_height from the video capture
    global frame_width, frame_height
    focal_length = frame_width  # Using frame width as an example focal length
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.double
    )

    # No lens distortion
    dist_coeffs = np.zeros((4, 1))

    # 3D model points of a generic face model
    object_points = np.array([
        (0.0, 0.0, 0.0),              # Nose tip
        (0.0, -330.0, -65.0),         # Chin
        (-225.0, 170.0, -135.0),       # Left eye corner
        (225.0, 170.0, -135.0),        # Right eye corner
        (-150.0, -150.0, -125.0),      # Left mouth corner
        (150.0, -150.0, -125.0)        # Right mouth corner
    ])

    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate Euler angles (pitch, yaw, roll)
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.array([x, y, z]) * 180 / np.pi  # Convert to degrees

def get_points(frame, frame_id, face_mesh_object, hands_object):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh_object.process(rgb_frame)
    hand_results = hands_object.process(rgb_frame)

    feature_points = {
        "left_eye": [],
        "right_eye": [],
        "left_pupil": [],
        "right_pupil": [],
        "mouth": [],
        "left_hand": [],
        "right_hand": [],
        "face_shape": [],
        "head_pose": None,
        "frame_id": frame_id,
    }

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

            mouth_landmarks = [
                face_landmarks.landmark[61], # left most point
                face_landmarks.landmark[291], # right most point
                face_landmarks.landmark[37], # top left point
                face_landmarks.landmark[267], # top right point
                face_landmarks.landmark[84], # bottom left point
                face_landmarks.landmark[314], # bottom right point
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
            face_shape_landmarks = [
                face_landmarks.landmark[10], # center of face
                face_landmarks.landmark[54], # left most point
                face_landmarks.landmark[284], # right most point
            ]

            for landmark in left_eye_landmarks:
                feature_points["left_eye"].append((landmark.x, landmark.y, landmark.z))

            for landmark in right_eye_landmarks:
                feature_points["right_eye"].append((landmark.x, landmark.y, landmark.z))

            for landmark in mouth_landmarks:
                feature_points["mouth"].append((landmark.x, landmark.y, landmark.z))
            
            for landmark in left_pupil_landmarks:
                feature_points["left_pupil"].append((landmark.x, landmark.y, landmark.z))

            for landmark in right_pupil_landmarks:
                feature_points["right_pupil"].append((landmark.x, landmark.y, landmark.z))

            for landmark in face_shape_landmarks:
                feature_points["face_shape"].append((landmark.x, landmark.y, landmark.z))

            feature_points["head_pose"] = calculate_head_pose(face_landmarks)

    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            handedness = hand_results.multi_handedness[idx].classification[0].label
            for landmark in hand_landmarks.landmark:
                if handedness == "Left":
                    feature_points["left_hand"].append((landmark.x, landmark.y, landmark.z))
                elif handedness == "Right":
                    feature_points["right_hand"].append((landmark.x, landmark.y, landmark.z))

    return feature_points

cap = cv2.VideoCapture("/home/harsh/Downloads/sem2/edgeai/edge ai project/trimmed_video.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_face_hands_points_function.avi', fourcc, fps_input, (frame_width, frame_height))

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    all_points = get_points(frame, frame_id, face_mesh, hands)

    # Draw facial and hand landmarks if available
    if all_points["left_eye"] or all_points["right_eye"] or all_points["face_shape"] or all_points["mouth"] or all_points["left_hand"] or all_points["right_hand"] or all_points["left_pupil"] or all_points["right_pupil"]:
        for x, y, z in all_points["left_eye"]:
            x_pixel, y_pixel = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)

        for x, y, z in all_points["right_eye"]:
            x_pixel, y_pixel = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)

        for x, y, z in all_points["mouth"]:
            x_pixel, y_pixel = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (255, 0, 0), -1)

        for x, y, z in all_points["left_hand"]:
            x_pixel, y_pixel = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 0, 255), -1)  # left hand points in red

        for x, y, z in all_points["right_hand"]:
            x_pixel, y_pixel = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 255), -1)  # right hand points in yellow
        for x, y, z in all_points["left_pupil"]:
            x_pixel, y_pixel = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (255, 0, 255), -1)
        for x, y, z in all_points["right_pupil"]:
            x_pixel, y_pixel = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (255, 0, 255), -1)
        for x, y, z in all_points["face_shape"]:
            x_pixel, y_pixel = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (255, 255, 0), -1)
        # Draw lines between face shape points

    
    # Optionally display head pose if available
    if all_points["head_pose"] is not None:
        pitch, yaw, roll = all_points["head_pose"]
        head_pose_text = "Pitch: {:.2f}, Yaw: {:.2f}, Roll: {:.2f}".format(pitch, yaw, roll)
        cv2.putText(frame, head_pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('Face and Hand Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()