import cv2

def trim_video(input_path, output_path, start_sec, end_sec):
    """
    Trim a video from start_sec to end_sec and save it to output_path
    
    Parameters:
    input_path (str): Path to the input video file
    output_path (str): Path to save the trimmed video
    start_sec (float): Start time in seconds
    end_sec (float): End time in seconds
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        out.write(frame)
        current_frame += 1
    
    cap.release()
    out.release()
    print(f"Trimmed video saved to {output_path}")

input_video = "/home/harsh/Downloads/sem2/edgeai/video.mp4"
output_video = "trimmed_video.mp4"
trim_video(input_video, output_video, 137, 197)