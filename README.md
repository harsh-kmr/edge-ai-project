# AI-Powered Multi-Class Pose Detection for Driver Behaviour Monitoring

## Overview

This project implements an AI-powered system for multi-class pose detection to monitor driver behavior in real-time. It utilizes computer vision techniques to analyze driver poses, eye movements, hand positions, and head orientation to classify various driver states such as distracted driving, fatigue, or normal driving.

The system is designed for edge computing environments, making it suitable for deployment on resource-constrained devices like automotive systems or mobile platforms.

## Features

- **Real-time Pose Detection**: Uses MediaPipe for efficient pose estimation
- **Multi-modal Feature Extraction**:
  - Eye tracking and blink detection
  - Hand gesture recognition
  - Head pose estimation
  - Phone detection (CNN-based)
- **Driver State Classification**: Machine learning models to classify driver behavior
- **Video Processing**: Support for both live camera feed and recorded video analysis
- **Optimized for Edge Devices**: Lightweight implementation suitable for edge AI deployment

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/harsh-kmr/edge-ai-project.git
   cd edge-ai-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

Key libraries used:
- **MediaPipe**: For pose and landmark detection
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations
- **PyTorch**: Deep learning framework (for custom models)
- **Matplotlib**: Visualization
- **Pandas**: Data manipulation

## Usage

### Running Inference on Recorded Video

```python
from inference_recorded import LiveInferencePipeline

# Initialize pipeline
pipeline = LiveInferencePipeline(
    video_source="path/to/video.mp4",
    output_path="output.mp4",
    save_video=True,
    display_output=True
)

# Run inference
pipeline.run()
```

### Real-time Inference

Modify the `video_source` to `0` for webcam or provide camera index.

### Custom Model Training

Use the provided scripts for data preprocessing and model training:
- `datacreator.py`: Data collection and preprocessing
- `preprocessing.py`: Feature extraction and cleaning
- `model.py`: Model definition and training

## Project Structure

```
├── augument.py              # Data augmentation scripts
├── base.py                  # Base feature extractor class
├── base_parallel.py         # Parallel processing utilities
├── datacreator.py           # Data collection and creation
├── eye.py                   # Eye feature extraction
├── hand.py                  # Hand feature extraction
├── hand_1.py                # Alternative hand detection
├── head_pose.py             # Head pose estimation
├── inference_recorded.py    # Main inference pipeline
├── model.py                 # ML model definitions
├── phone_detector.py        # Phone detection CNN
├── phone_predict.py         # Phone prediction utilities
├── pose_estimation_model.py # Pose estimation model
├── preprocessing.py         # Data preprocessing
├── sample_video.py          # Video sampling utilities
├── tools.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── testing.ipynb            # Jupyter notebook for testing
├── pose_estimate_model.ipynb # Pose model notebook
└── demo1.mp4               # Demo video
```

## Team Members and Contributions

### Stage 1: Feasibility and Initial Setup
- **Harsh Kumar, Pranav Tiwari**: Conducted feasibility tests and evaluated libraries (MediaPipe, Dlib, etc.) for pose detection
- **Harsh Kumar**: Selected features and labels for driver state classification
- **Harsh Kumar**: Defined data collection rules and standardized the process

### Stage 2: Data Collection
- **Harsh Kumar, Pranav Tiwari, Priyanka Nihalchandani, Vikas Kumar**: Collected driver behavior data across multiple sessions
- Special thanks to **Prof. Pandarasamy Arjunan** and **Prof. Punit Rathore** for providing access to the driving simulator
- **Harsh Kumar**: Performed manual labeling of the dataset

### Stage 3: Pipeline Development
- **Harsh Kumar**: Developed data preprocessing and cleaning scripts (`datacreator.py`, `preprocessing.py`), `base.py`, and `augument.py`
- **Eye Feature Extraction**: Harsh Kumar and Pranav Tiwari developed the module. Debugging and correction by Harsh Kumar; optimization by Pranav Tiwari
- **Hand Feature Extraction**: Developed by Priyanka Nihalchandani. Debugging and correction by Harsh Kumar; optimization by Pranav Tiwari
- **Head Feature Extraction**: Developed by Vikas Kumar. Debugging and correction by Harsh Kumar; optimization by Pranav Tiwari
- **Harsh Kumar**: Implemented `phone_detector.py` (later dropped due to resource constraints). CNN model to detect phone, created dataset, trained model
- **Vikas Kumar, Harsh Kumar**: Developed `phone_predict.py`

### Stage 4: Model Training, Inference, and Optimization
- **Pranav Tiwari**: Developed `tools.py` and `base_parallel.py` (with contributions from Vikas Kumar)
- **Harsh Kumar, Pranav Tiwari**: Developed `inference_recorded.py` for real-time and recorded video inference
- **Harsh Kumar**: Conducted model training and performance evaluation

### Stage 5: Report and Presentation
- **Vikas Kumar, Priyanka Nihalchandani**: Drafted the initial project report
- **Harsh Kumar**: Edited and finalized the project report
- **Priyanka Nihalchandani**: Created the initial presentation slides
- **Harsh Kumar**: Revised and finalized the presentation

### Stage 6: Pose Detection Model (PyTorch)
- **Harsh Kumar**: Created the custom pose estimation dataset and trained the pose model
- **Pranav Tiwari**: Developed the PyTorch data loader for the pose model

## Links

- **GitHub Repository**: [https://github.com/harsh-kmr/edge-ai/tree/main](https://github.com/harsh-kmr/edge-ai/tree/main)
- **Demo Video**: [https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/harshkumar1_iisc_ac_in/Eo_0ujCKjClPmhYjkk5YbucBN1_mPig8EBLInQfk-A8XeA?e=dUtcjh](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/harshkumar1_iisc_ac_in/Eo_0ujCKjClPmhYjkk5YbucBN1_mPig8EBLInQfk-A8XeA?e=dUtcjh)
- **Dataset**: Shared via email with project supervisor
- **Model Files**: [https://github.com/harsh-kmr/edge-ai/blob/main/model.py](https://github.com/harsh-kmr/edge-ai/blob/main/model.py)

## License

This project is developed for academic purposes. Please refer to the repository for any licensing information.

## Contact

For questions or contributions, please contact the team members or create an issue in the GitHub repository.
