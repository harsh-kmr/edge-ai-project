# base.py

"""
Defines the standard structure and conventions for feature extraction modules.

Goal:
To allow us to work on different feature groups (e.g., eye features,
hand features, head pose features) in separate files (like eye.py, hand.py)
and later merge them easily into a single comprehensive class.

Convention for Feature Modules (e.g., eye.py, hand.py):

1.  **Single Class:** Each module should contain a primary class responsible
    for calculating its group of features (e.g., `EyeFeatures`, `HandFeatures`).
    Follow the pattern shown in `eye.py`'s `Posepoints` class for consistency.

2.  **Initialization (`__init__`)**:
    - Accepts necessary MediaPipe detector objects (e.g., FaceMesh, Hands, Pose)
      or other primary data sources.
    - Initializes an empty pandas DataFrame: `self.df = pd.DataFrame()` if not given. This
      DataFrame will store calculated features frame-by-frame and serve as
      a simple state/history mechanism for temporal features.
    - Stores the detector objects (e.g., `self.face_mesh_object = face_mesh_object`).

3.  **Core Processing Method (`process_frame`)**:
    - Signature: `process_frame(self, frame, frame_id)`
    - Responsibilities:
        - Run the relevant MediaPipe detectors on the input `frame`
          (e.g., `face_results = self.face_mesh_object.process(frame)`).
        - Store the current `frame_id` (e.g., `self.frame_id = frame_id`).
        - Call a method to calculate and save the features for the current frame
          (e.g., `self.calculate_and_save_features(face_results, hand_results)`
          then send the results to the `save_to_df` method).
        - Optionally return the raw detector results if needed downstream.

4.  **Feature Calculation & Saving (`calculate_and_save_features` or similar)**:
    - This logic (demonstrated within `save_to_df` in `eye.py`) should:
        - Call helper methods to extract necessary landmarks/data points from
          detector results (e.g., `self.get_eye_landmarks(face_results)`).
        - Call individual methods to calculate each feature belonging to this
          module (e.g., `self.EyeAspectRatio2D(...)`, `self.no_visible_eyes(...)`).
        - Aggregate these calculated features into a dictionary or pandas Series,
          including the `frame_id`.
        - Append this new row of data to the internal DataFrame:
          `self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)`

5.  **Individual Feature Methods**:
    - Each distinct feature (like EAR, MAR, Hand Distance) should have its own
      calculation method (e.g., `EyeAspectRatio2D`, `variance_pupil_movement`).
    - Use `@staticmethod` for methods that are pure functions (only depend on
      their input arguments).
    - Use instance methods (with `self`) for features that require state or
      access to historical data stored in `self.df`.
    - **Temporal Features**: For features requiring data from previous frames
      (like variance, counts over windows, duration), access the necessary history
      directly from `self.df`. *Note: Reading from the DataFrame (or potentially
      a CSV via the DataFrame) on each frame for temporal calculations is acceptable
      for this POC, despite potential inefficiencies.*

6.  **Data Extraction Helpers**:
    - Include methods (often static) dedicated to extracting specific sets of
      landmarks or data points from the raw detector results (e.g.,
      `get_eye_landmarks`).

7.  **Saving to CSV (`save_to_csv`)**:
    - Provide a method `save_to_csv(self, csv_path)` that saves the entire
      `self.df` DataFrame to the specified path. This persists the results.
    - Store the `csv_path`

Adhering to this structure (consistent method names like `__init__`,
`process_frame`, `save_to_csv`, the use of `self.df`, and clear separation
of feature calculations) will make the final integration process significantly
easier, primarily involving copying the feature calculation methods, data
extraction helpers, and the relevant lines from the `calculate_and_save_features`
(or `save_to_df`) logic into the final unified class.
"""

# Potential imports needed by multiple modules can be placed here,
# but it's often cleaner to keep imports within the specific files that use them.
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2

# Placeholder for the final merged class structure (or potentially an abstract base class later)
# class MasterFeatureExtractor:
#     def __init__(self, ...):
#         # Initialize all required detectors
#         # Initialize self.df
#         pass
#
#     def process_frame(self, frame, frame_id):
#         # Run all detectors
#         # Call combined feature calculation/saving
#         pass
#
#     def calculate_and_save_all_features(self, ...):
#         # Call methods for eye features
#         # Call methods for hand features
#         # Call methods for head features
#         # ... etc ...
#         # Aggregate all features into data dictionary
#         # Append to self.df
#         pass
#
#     # --- Eye Feature Methods (Copied from eye.py) ---
#     @staticmethod
#     def get_eye_landmarks(...): ...
#     @staticmethod
#     def EyeAspectRatio2D(...): ...
#     # ... all other eye methods ...
#
#     # --- Hand Feature Methods (Copied from hand.py) ---
#     # ...
#
#     # --- Head Feature Methods (Copied from head.py) ---
#     # ...
#
#     # --- Etc. ---
#
#     def save_to_csv(self, csv_path):
#         self.df.to_csv(csv_path, index=False)
#         print(f"Comprehensive data saved to {csv_path}.")