import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random

def get_metrics_classification(y_true, y_pred, device=None):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    return metrics

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0005, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_loss = float('inf')  
        self.last_loss = float('inf')
        
        # Create directory for checkpoint if it doesn't exist
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else '.', exist_ok=True)

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:  
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        self.last_loss = val_loss

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            if self.last_loss == float('inf'):
                self.trace_func(f'First save with Validation loss --> {val_loss:.6f}. Saving model to {self.path}...')
            else:
                self.trace_func(f'Validation loss decreased ({self.last_loss:.6f} --> {val_loss:.6f}). Saving model to {self.path}...')
        torch.save(model.state_dict(), self.path)

class CNN_trainer():
    def __init__(self, model, train_loader, val_loader, num_epochs=10,
                 patience=5, verbose=True, trace_func=print, delta=0.0005,
                 optimizer=None, loss_fn=None, l1_lambda=0.0, checkpoint_path="checkpoint.pt"):
        
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.num_epochs = num_epochs
        self.patience = patience
        self.verbose = verbose
        self.trace_func = trace_func

        self.early_stopping = EarlyStopping(patience=patience, verbose=verbose, trace_func=trace_func,
                                            path=checkpoint_path, delta=delta)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train_step(self):
        running_loss = 0.0
        all_preds = []
        all_labels = []
        self.model.train()
        
        for images, labels in (tqdm(self.train_loader) if self.verbose else self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            outputs = outputs.squeeze(1)
            loss = self.loss_fn(outputs, labels)
            
            if self.l1_lambda > 0:
                l1_norm = sum(torch.abs(param).sum() for param in self.model.parameters())
                loss += self.l1_lambda * l1_norm
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
        
        avg_loss = running_loss / len(self.train_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = get_metrics_classification(all_labels, all_preds, self.device)
        
        return avg_loss, metrics
    
    def eval_step(self, data_loader=None):
        if data_loader is None:
            data_loader = self.val_loader
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                outputs = outputs.squeeze(1)
                loss = self.loss_fn(outputs, labels)
                
                running_loss += loss.item()
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = running_loss / len(data_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = get_metrics_classification(all_labels, all_preds, self.device)
        
        return avg_loss, metrics, all_preds, all_labels
    
    def fit(self):
        self.model.to(self.device)
        if self.verbose:
            print(f"Training model on {self.device}")
        
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []

        for epoch in range(self.num_epochs):
            train_loss, train_metric = self.train_step()
            train_losses.append(train_loss)
            train_metrics.append(train_metric)
            
            if self.verbose:
                self.trace_func(f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {train_loss:.4f}, "
                f"Training Accuracy: {train_metric['accuracy']:.4f}")
            
            val_loss, val_metric, _, _ = self.eval_step()
            val_losses.append(val_loss)
            val_metrics.append(val_metric)
            
            if self.verbose:
                self.trace_func(f"Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_metric['accuracy']:.4f}")
            
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                if self.verbose:
                    self.trace_func("Early stopping")
                break
        
        # Load the best model
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        
        _, train_metric_final, train_preds, train_labels_all = self.eval_step(self.train_loader)
        _, val_metric_final, val_preds, val_labels_all = self.eval_step()
        
        self.trace_func("Training metrics:")
        self.trace_func(train_metric_final)
        self.trace_func("Validation metrics:")
        self.trace_func(val_metric_final)
        
        return self.model, train_losses, val_losses, train_metric_final, val_metric_final


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


class PhoneDataset(Dataset):
    def __init__(self, phone_dir, no_phone_dir, transform=None):
        self.phone_images = glob.glob(os.path.join(phone_dir, "*.jpg"))
        self.no_phone_images = glob.glob(os.path.join(no_phone_dir, "*.jpg"))
        self.all_images = [(img, 1) for img in self.phone_images] + [(img, 0) for img in self.no_phone_images]
        random.shuffle(self.all_images)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),  # Resize larger than final size for cropping
            transforms.RandomCrop(224),     # Random crop for position variety
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flips
            transforms.RandomRotation(15),  # Slight rotations
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Color variations
            transforms.RandomAutocontrast(p=0.2),  # Random autocontrast
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.all_images)
        
    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder or skip the sample
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float)


# def extract_frames_from_videos(pwd, csv_path, phone_dir="phone", no_phone_dir="no_phone", phone_fraction=0.4, seed=42):
#     """Extract frames from videos and save them to respective directories"""
#     # Create directories
#     phone_dir  = os.path.join(pwd, phone_dir)
#     no_phone_dir = os.path.join(pwd, no_phone_dir)
#     csv_path = os.path.join(pwd, csv_path)

#     os.makedirs(phone_dir, exist_ok=True)
#     os.makedirs(no_phone_dir, exist_ok=True)
    
#     # Read dataset
#     df = pd.read_csv(csv_path)
    
#     # Get phone and no phone samples
#     phone_df = df[df['label'] == 'phone']
#     phone_df = phone_df.sample(frac=phone_fraction, random_state=seed)
#     not_phone_df = df[df['label'] != 'phone']
#     not_phone_df = not_phone_df.sample(n=len(phone_df), random_state=seed)
    
#     # Save frames from videos
#     for idx, row in tqdm(phone_df.iterrows(), desc="Saving phone frames"):
#         try:
#             video_path = row['video_address']
#             video_path = os.path.join(pwd, video_path)
#             cap = cv2.VideoCapture(video_path)
#             frame_count = 0
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 save_path = os.path.join(phone_dir, f"phone_{idx}_frame{frame_count}.jpg")
#                 cv2.imwrite(save_path, frame)
#                 frame_count += 1
#             cap.release()
#         except Exception as e:
#             print(f"Error processing video {row['video_address']}: {e}")
    
#     for idx, row in tqdm(not_phone_df.iterrows(), desc="Saving no phone frames"):
#         try:
#             video_path = row['video_address']
#             video_path = os.path.join(pwd, video_path)
#             cap = cv2.VideoCapture(video_path)
#             frame_count = 0
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 save_path = os.path.join(no_phone_dir, f"no_phone_{idx}_frame{frame_count}.jpg")
#                 cv2.imwrite(save_path, frame)
#                 frame_count += 1
#             cap.release()
#         except Exception as e:
#             print(f"Error processing video {row['video_address']}: {e}")



def extract_frames_from_videos(
    pwd,
    csv_path,
    phone_dir="phone",
    no_phone_dir="no_phone",
    frame_fraction=0.35,
    seed=42
):
    """Extract a random subset of frames (with probability=frame_fraction) from each video,
    saving phone vs. no-phone frames into balanced directories."""
    # Build absolute paths
    phone_dir = os.path.join(pwd, phone_dir)
    no_phone_dir = os.path.join(pwd, no_phone_dir)
    csv_path  = os.path.join(pwd, csv_path)

    # Make sure output dirs exist
    os.makedirs(phone_dir, exist_ok=True)
    os.makedirs(no_phone_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)

    # All phone videos
    phone_videos = df[df['label'] == 'phone'].reset_index(drop=True)
    # Randomly pick same number of non-phone videos
    non_phone_videos = (
        df[df['label'] != 'phone']
        .sample(n=len(phone_videos), random_state=seed)
        .reset_index(drop=True)
    )

    # Set up RNG
    random.seed(seed)

    def _save_sampled_frames(videos_df, out_dir, prefix):
        for vid_idx, row in tqdm(
            videos_df.iterrows(),
            total=len(videos_df),
            desc=f"Saving {prefix} frames"
        ):
            video_path = os.path.join(pwd, row['video_address'])
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # sample this frame?
                    if random.random() < frame_fraction:
                        fname = f"{prefix}_{vid_idx}_frame{frame_count}.jpg"
                        cv2.imwrite(os.path.join(out_dir, fname), frame)
                    frame_count += 1
                cap.release()
            except Exception as e:
                print(f"Error processing {row['video_address']}: {e}")

    # Extract & save
    _save_sampled_frames(phone_videos,     phone_dir,     "phone")
    _save_sampled_frames(non_phone_videos, no_phone_dir, "no_phone")



def train_model(pwd,phone_dir="phone", no_phone_dir="no_phone", batch_size=32, num_epochs=20, 
                patience=5, checkpoint_path="phone_detector_model.pt"):
    """Train the phone detection model"""

    phone_dir = os.path.join(pwd, phone_dir)
    no_phone_dir = os.path.join(pwd, no_phone_dir)

    # Create dataset
    dataset = PhoneDataset(phone_dir, no_phone_dir)
    print(f"Dataset created with {len(dataset)} images")

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model, optimizer, and loss function
    model = PhoneDetectorModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    # Train the model
    trainer = CNN_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        patience=patience,
        verbose=True,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_path=checkpoint_path
    )

    trained_model, train_losses, val_losses, train_metrics, val_metrics = trainer.fit()

    # Save the trained model
    final_path = checkpoint_path.replace(".pt", "_final.pt")
    torch.save(trained_model.state_dict(), final_path)
    print(f"Model saved to {final_path}")

    return trained_model, train_metrics, val_metrics


if __name__ == "__main__":
    # Set paths
    pwd = "/home/harsh/Downloads/sem2/edgeai/edge ai project/dummy data/cleaned data"
    csv_path = "video_labels.csv"  # Change this to your CSV path
    
    # Extract frames from videos
    extract_frames_from_videos(pwd= pwd, csv_path=csv_path)
    
    # Train model
    model, train_metrics, val_metrics = train_model(pwd=pwd)