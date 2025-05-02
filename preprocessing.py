import cv2
import numpy as np
import os
from dataclasses import dataclass
#from tqdm import tqdm
import tempfile
import traceback



@dataclass
class Preprocessing:
    image_size: tuple = (224, 224)
    color_mode: str = 'rgb'
    video_fps: int = 30
    video_duration: int = 30
    video_dir: str = None
    video_output_dir: str = None
    video_file_type: str = '.mp4'
    convert_color_flag: bool = True
    save_video_flag: bool = False
    debug_flag: bool = False


    def _get_video_files_(self):
        if self.debug_flag:
            print(f"DEBUG: Searching for video files in {self.video_dir}")
        if self.video_dir is None:
            raise ValueError("video_dir must be set to a valid directory.")
        video_files = sorted(f for f in os.listdir(self.video_dir)
                     if f.endswith(self.video_file_type))
        if not video_files:
            raise ValueError(f"No video files found in {self.video_dir} with extension {self.video_file_type}.")
        if self.debug_flag:
            print(f"DEBUG: Found {len(video_files)} videos")
        return video_files
    
    def _load_video_(self, video_file):
        video_path = os.path.join(self.video_dir, video_file)
        
        if self.debug_flag:
            print(f"DEBUG: Loading video from {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.debug_flag:
            print(f"DEBUG: Video dimensions: {frame_width}x{frame_height}")
        
        if frame_width < frame_height:
            if self.debug_flag:
                print(f"DEBUG: Width ({frame_width}) > Height ({frame_height}), rotating video")
            # Creating a rotation matrix
            # We'll rotate frames individually during processing
            self.rotate_video = True
        else:
            self.rotate_video = False

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        if self.debug_flag:
            print(f"DEBUG: Video loaded successfully with properties: FPS={cap.get(cv2.CAP_PROP_FPS)}, Frames={cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        return cap
    
    def _resize_frame_(self, frame):
        if self.debug_flag and frame is not None:
            print(f"DEBUG: Processing frame with shape {frame.shape[:2]}")
        
        # Check if rotation is needed (width > height)
        if hasattr(self, 'rotate_video') and self.rotate_video:
            if self.debug_flag:
                print(f"DEBUG: Rotating frame")
            # Rotate 90 degrees counterclockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        if self.debug_flag and frame is not None:
            print(f"DEBUG: Resizing frame from {frame.shape[:2]} to {self.image_size}")
        
        # Resize the frame
        frame = cv2.resize(frame, self.image_size)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame
    
    def _convert_color_(self, frame):
        if self.debug_flag:
            print(f"DEBUG: Converting color to {self.color_mode}")
        if self.color_mode == 'rgb':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.color_mode == 'gray':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported color mode: {self.color_mode}")
    
    def _change_fps_(self, cap):
        current_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if self.debug_flag:
            print(f"DEBUG: Current video FPS: {current_fps}, Target FPS: {self.video_fps}")
        
        if current_fps == self.video_fps:
            if self.debug_flag:
                print("DEBUG: FPS already matches target, no change needed")
            return cap, None  # No change needed
            
        # If FPS needs to be changed, we need to read all frames and create a new video
        if self.debug_flag:
            print("DEBUG: FPS change required, reading all frames...")
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError("No frames were extracted from the video")
            
        if self.debug_flag:
            print(f"DEBUG: Read {len(frames)} frames, creating temporary video with new FPS")
            
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, self.video_fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        if self.debug_flag:
            print(f"DEBUG: Created temporary video at {temp_path}")
            
        new_cap = cv2.VideoCapture(temp_path)
        if not new_cap.isOpened():
            raise ValueError(f"Could not create new video with adjusted FPS")
            
        if self.debug_flag:
            print(f"DEBUG: New video created with FPS: {new_cap.get(cv2.CAP_PROP_FPS)}")
            
        return new_cap, temp_path
    
    def _trim_video_(self, cap):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        required_frames = self.video_duration * fps
        
        if self.debug_flag:
            print(f"DEBUG: Total frames: {total_frames}, Required frames: {required_frames}")
        
        if total_frames <= required_frames:
            if self.debug_flag:
                print("DEBUG: Video shorter than or equal to required duration, using all frames")
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            return frames
        else:
            excess_frames = total_frames - required_frames
            start_frame = excess_frames // 2
            end_frame = total_frames - (excess_frames - start_frame)
            
            if self.debug_flag:
                print(f"DEBUG: Trimming video: starting at frame {start_frame}, ending at frame {end_frame}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            for i in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    if self.debug_flag:
                        print(f"DEBUG: Failed to read frame {i}")
                    break
                frames.append(frame)
            
            if self.debug_flag:
                print(f"DEBUG: Trimmed video to {len(frames)} frames")
            
            return frames
    
    
    def _save_video_(self, frames, video_file, fps=None):
        if self.debug_flag:
            print(f"DEBUG: Saving processed video to {self.video_output_dir}/{video_file}")
            
        if self.video_output_dir is None:
            raise ValueError("video_output_dir must be set to a valid directory.")
        
        if not os.path.exists(self.video_output_dir):
            if self.debug_flag:
                print(f"DEBUG: Creating output directory {self.video_output_dir}")
            os.makedirs(self.video_output_dir)
            
        output_path = os.path.join(self.video_output_dir, video_file)

        if not frames:
            raise ValueError("No frames to save")


        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = fps or self.video_fps  
        
        if self.debug_flag:
            print(f"DEBUG: Creating video writer with dimensions {width}x{height}, FPS: {fps}")
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i, frame in enumerate(frames):
            if self.color_mode == 'rgb':
                if self.debug_flag and i == 0:
                    print("DEBUG: Converting RGB frames back to BGR for saving")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif self.color_mode == 'gray':
                if self.debug_flag and i == 0:
                    print("DEBUG: Converting grayscale frames to BGR for saving")
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)

        out.release()
        print(f"Saved processed video to: {output_path}")

        
    def __getitem__(self, idx):
        if self.debug_flag:
            print(f"DEBUG: Processing video at index {idx}")
            
        video_files = self._get_video_files_()
        video_file = video_files[idx]
        
        if self.debug_flag:
            print(f"DEBUG: Video file: {video_file}")
            
        cap = self._load_video_(video_file)
        cap, tempfilepath = self._change_fps_(cap)
        
        frames = self._trim_video_(cap)
        
        if self.debug_flag:
            print(f"DEBUG: Processing {len(frames)} frames")
            
        processed_frames = []
        for i, frame in enumerate(frames):
            if self.debug_flag and i % 100 == 0:
                print(f"DEBUG: Processing frame {i}/{len(frames)}")
                
            if self.convert_color_flag:
                frame = self._convert_color_(frame)
            frame = self._resize_frame_(frame)
            processed_frames.append(frame)
        
        if not video_file.endswith('.mp4'):
            if self.debug_flag:
                print(f"DEBUG: Changing output file extension from {self.video_file_type} to .mp4")
            video_file = video_file.replace(self.video_file_type, '.mp4')
        
        if self.save_video_flag:
            if self.debug_flag:
                print(f"DEBUG: Saving processed video")
            self._save_video_(processed_frames, video_file)
            
        cap.release()
        if tempfilepath and os.path.exists(tempfilepath):
            if self.debug_flag:
                print(f"DEBUG: Removing temporary file {tempfilepath}")
            os.remove(tempfilepath)
            
        if self.debug_flag:
            print(f"DEBUG: Finished processing video {idx}")
            
        return processed_frames
    
    def __len__(self):
        video_files = self._get_video_files_()
        return len(video_files)
    
    def process_videos(self):
        video_files = self._get_video_files_()
        if self.debug_flag:
            print(f"DEBUG: Processing {len(video_files)} videos")
            
        for idx, video_file in enumerate(video_files):
            print(f"Processing video {idx + 1}/{len(video_files)}: {video_file}")
            try:
                self.__getitem__(idx)
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
                if self.debug_flag:
                    print(f"DEBUG: Exception details:\n{traceback.format_exc()}")
                
    def __call__(self):
        if self.debug_flag:
            print(f"DEBUG: Starting video processing with settings: image_size={self.image_size}, color_mode={self.color_mode}, fps={self.video_fps}, duration={self.video_duration}")
        self.process_videos()

if __name__ == "__main__":
    video_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project"
    video_output_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/output_videos"
    
    processor = Preprocessing(
        image_size=(256, 256),
        color_mode='rgb',
        video_fps=30,
        video_duration=30,
        video_dir=video_dir,
        video_output_dir=video_output_dir,
        save_video_flag=True
    )
    
    processor()

            
