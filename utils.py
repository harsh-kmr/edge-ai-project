import numpy as np

def euclidean_distance_2d(point1, point2):
    """Calculates the Euclidean distance between two 2D points."""
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def euclidean_distance_3d(point1, point2):
    """Calculates the Euclidean distance between two 3D points."""
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def EyeAspectRatio2D(eye):
     # Formula : dist(p2,p6) + dist(p3,p5) / 2*dist(p1, p4)
    p1 = (eye[0][0], eye[0][1], eye[0][2])
    p2 = (eye[2][0], eye[2][1], eye[2][2])
    p3 = (eye[3][0], eye[3][1], eye[3][2])
    p4 = (eye[1][0], eye[1][1], eye[1][2])
    p5 = (eye[5][0], eye[5][1], eye[5][2])
    p6 = (eye[4][0], eye[4][1], eye[4][2])

    dist_p2_p6 = euclidean_distance_2d(p2, p6)
    dist_p3_p5 = euclidean_distance_2d(p3, p5)
    dist_p1_p4 = euclidean_distance_2d(p1, p4)

    if dist_p1_p4 == 0:
        return 0  # Avoid division by zero

    return (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)

def EyeAspectRatio3D(eye):
    # Formula : dist(p2,p6) + dist(p3,p5) / 2*dist(p1, p4)
    p1 = (eye[0][0], eye[0][1], eye[0][2])
    p2 = (eye[2][0], eye[2][1], eye[2][2])
    p3 = (eye[3][0], eye[3][1], eye[3][2])
    p4 = (eye[1][0], eye[1][1], eye[1][2])
    p5 = (eye[5][0], eye[5][1], eye[5][2])
    p6 = (eye[4][0], eye[4][1], eye[4][2])

    dist_p2_p6 = euclidean_distance_3d(p2, p6)
    dist_p3_p5 = euclidean_distance_3d(p3, p5)
    dist_p1_p4 = euclidean_distance_3d(p1, p4)

    if dist_p1_p4 == 0:
        return 0  # Avoid division by zero

    return (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)

def center_pupil(pupil):
    """Calculates the center of the pupil given its landmarks."""
    if not pupil:
        return None
    
    x_coords = [p[0] for p in pupil]
    y_coords = [p[1] for p in pupil]
    
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    
    return (center_x, center_y)

def center_eye(eye):
    """Calculates the center of the eye given its landmarks."""
    if not eye:
        return None

    x_coords = [p[0] for p in eye]
    y_coords = [p[1] for p in eye]

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    return (center_x, center_y)