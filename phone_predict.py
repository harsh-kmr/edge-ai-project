import torch
import os
import glob
import cv2
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ========== Define transformation for input frames ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== Load trained model ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = PhoneDetectionModel()
# model.load_state_dict(torch.load("phone_detector_model_final.pt", map_location=device))
# model.to(device)
# model.eval()

# ========== Function to process a single video =============
def predict_video(video_path, model, transform, device):
    predictions = []
    video_name = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame (OpenCV BGR) to PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = transform(image).unsqueeze(0).to(device)

            # Predict
            output = model(image)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float().item()  # 0 or 1

            predictions.append({
                "video_name": video_name,
                "frame_id": frame_id,
                "phone_present": int(pred)
            })

            frame_id += 1

    cap.release()
    return predictions

# ========== Main code ==========
if __name__ == "__main__":
    # INPUT: folder containing videos or single video
    input_path = "input_videos"  # folder or single video file
    output_csv = "phone_predictions.csv"

    all_predictions = []

    if os.path.isdir(input_path):
        video_files = glob.glob(os.path.join(input_path, "*.mp4")) + \
                      glob.glob(os.path.join(input_path, "*.avi")) + \
                      glob.glob(os.path.join(input_path, "*.mov"))
    else:
        video_files = [input_path]

    for video in tqdm(video_files, desc="Processing videos"):
        preds = predict_video(video, model, transform, device)
        all_predictions.extend(preds)

    # Save all predictions to CSV
    df = pd.DataFrame(all_predictions)
    df.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")