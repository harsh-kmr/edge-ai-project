from torch import nn
from collections import deque
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

class PhoneDetectorModel(nn.Module):
    def __init__(self):
        super(PhoneDetectorModel, self).__init__()
        self.enc_conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.enc_conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.enc_conv3 = nn.Conv2d(32, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.enc_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.enc_conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(1024)
        
        self.relu = nn.ReLU()

        self.linear_1 = nn.LazyLinear(256)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear_2 = nn.Linear(256, 1)
        
    def forward(self, X):
        X = self.relu(self.bn1(self.enc_conv1(X)))
        X = self.relu(self.bn2(self.enc_conv2(X)))
        X = self.relu(self.bn3(self.enc_conv3(X)))
        X = self.relu(self.bn4(self.enc_conv4(X)))  # Added ReLU activation here
        X = self.relu(self.bn5(self.enc_conv5(X)))
        X = self.relu(self.bn6(self.enc_conv6(X)))  # Added ReLU activation here

        X = X.flatten(start_dim=1)

        X = self.relu(self.bn7(self.linear_1(X)))
        X = self.linear_2(X)
        return X


class phone_detector():
    def __init__(self, model_path, phone_window_size=30):
        self.model = PhoneDetectorModel()
        self.model.load_state_dict(torch.load(model_path))
        self.phone_window = deque(maxlen=phone_window_size)
    
    def detect_phone(self, frame):
        image = Image.fromarray(frame)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
        self.model.eval()

        with torch.no_grad():
            output = self.model(image)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float().item()
        self.phone_window.append(int(pred))
        return int(pred)
    
    def phone_duration(self):
        if len(self.phone_window) < 30:
            return 0
        phone_count = sum(self.phone_window)
        return phone_count / len(self.phone_window)
    
    def calculate_phone_feature(self, frame, frame_id):
    
        phone_feature = self.detect_phone(frame)
        phone_duration = self.phone_duration()
        return {"phone_presence": phone_feature, "phone_duration": phone_duration}


if __name__ == "__main__":

    df = pd.DataFrame(columns=["frame_id", "phone_presence", "phone_duration"])

    def process_video(video_path, model_path):
        detector = phone_detector(model_path)
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            phone_feature = detector.calculate_phone_feature(frame, frame_id)
            df.loc[frame_id] = [frame_id, phone_feature["phone_presence"], phone_feature["phone_duration"]]
            frame_id += 1

        cap.release()
        return df
    
    process_video("/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/phone_20.mp4",
               "/home/harsh/Downloads/sem2/edgeai/edge ai project/phone_detector_model_final.pt")
    
    print(df)

    df  = pd.DataFrame( columns=["frame_id", "phone_presence", "phone_duration"])

    process_video("/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data/normal_5.mp4",
               "/home/harsh/Downloads/sem2/edgeai/edge ai project/phone_detector_model_final.pt")

    print(df)