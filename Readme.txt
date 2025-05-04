AI-Powered Multi-Class Pose Detection for Driver Behaviour Monitoring

Team Members and Individual Contributions

Stage 1: Feasibility and Initial Setup
- Harsh Kumar, Pranav Tiwari: Conducted feasibility tests and evaluated libraries (MediaPipe, Dlib, etc.) for pose detection.
- Harsh Kumar: Selected features and labels for driver state classification.
- Harsh Kumar: Defined data collection rules and standardized the process.

Stage 2: Data Collection
- Harsh Kumar, Pranav Tiwari, Priyanka Nihalchandani, Vikas Kumar: Collected driver behavior data across multiple sessions.
- Special thanks to Prof. Pandarasamy Arjunan and Prof. Punit Rathore for providing access to the driving simulator.
- Harsh Kumar: Performed manual labeling of the dataset.

Stage 3: Pipeline Development
- Harsh Kumar: Developed data preprocessing and cleaning scripts (Datacreator.py, preprocessing.py), base.py, and augment.py.
- Eye Feature Extraction: Harsh Kumar and Pranav Tiwari developed the module. Debugging and correction by Harsh Kumar; optimization by Pranav Tiwari.
- Hand Feature Extraction: Developed by Priyanka Nihalchandani. Debugging and correction by Harsh Kumar; optimization by Pranav Tiwari.
- Head Feature Extraction: Developed by Vikas Kumar. Debugging and correction by Harsh Kumar; optimization by Pranav Tiwari.
- Harsh Kumar: Implemented phone_detector.py (later dropped due to resource constraints). CNN model to detect phone, created dataset , trained Model
- Vikas Kumar, Harsh Kumar: Developed phone_predict.py.

Stage 4: Model Training, Inference, and Optimization
- Pranav Tiwari: Developed tools.py and base_parallel.py (with contributions from Vikas Kumar).
- Harsh Kumar, Pranav Tiwari: Developed inference_recorded.py for real-time and recorded video inference.
- Harsh Kumar: Conducted model training and performance evaluation.

Stage 5: Report and Presentation
- Vikas Kumar, Priyanka Nihalchandani: Drafted the initial project report.
- Harsh Kumar: Edited and finalized the project report.
- Priyanka Nihalchandani: Created the initial presentation slides.
- Harsh Kumar: Revised and finalized the presentation.

Stage 6: Pose Detection Model (PyTorch)
- Harsh Kumar: Created the custom pose estimation dataset and trained the pose model.
- Pranav Tiwari: Developed the PyTorch data loader for the pose model.

GitHub Repository: https://github.com/harsh-kmr/edge-ai/tree/main
