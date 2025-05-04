import cv2
import numpy as np
import random
from dataclasses import dataclass, field

# Import the Preprocessing class from the original file
from preprocessing import Preprocessing

@dataclass
class Augment(Preprocessing):
    rotation_range: tuple = (-10, 10)  
    shift_range: float = 0.05  
    scale_range: tuple = (0.9, 1.1)  
    mirror_prob: float = 0.5 
    
    enable_augmentation: bool = True
    
    def _apply_augmentations_(self, frame):
        """Apply random augmentations to a single frame"""
        if not self.enable_augmentation:
            return frame
            
        height, width = frame.shape[:2]
        
        center = (width // 2, height // 2)
        
        # 1. Random rotation
        angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        
        # 2. Random scale
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 3. Random shift
        tx = random.uniform(-self.shift_range, self.shift_range) * width
        ty = random.uniform(-self.shift_range, self.shift_range) * height
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply affine transformation
        frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)
            
        return frame
    def mirror(self, frame):
        return cv2.flip(frame, 1) 
    
    def __getitem__(self, idx):
        frames = super().__getitem__(idx)
    
        if self.enable_augmentation:
            augmented_frames = []
            mirror_flag = random.random() > self.mirror_prob
            for frame in frames:
                augmented_frame = self._apply_augmentations_(frame)
                if mirror_flag:
                    augmented_frame = self.mirror(augmented_frame)
                augmented_frames.append(augmented_frame)
            return augmented_frames

        else:
            return frames
            
    def process_videos(self):
        video_files = self._get_video_files_()
        for idx, video_file in enumerate(video_files):
            print(f"Processing video {idx + 1}/{len(video_files)}: {video_file} with augmentations")
            try:
                self.__getitem__(idx)
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")


if __name__ == "__main__":
    video_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project"
    video_output_dir = "/home/harsh/Downloads/sem2/edgeai/edge ai project/output_videos"
    
    augmentor = Augment(
        image_size=(224, 224),
        color_mode='rgb',
        video_fps=30,
        video_duration=30,
        video_dir=video_dir,
        video_output_dir=video_output_dir,
        save_video_flag=True,
        enable_augmentation=True
    )
    
    augmentor()