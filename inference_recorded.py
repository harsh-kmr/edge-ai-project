import cv2
import numpy as np
import os
import time
import traceback
from dataclasses import dataclass
from base2 import MasterFeatureExtractor
import mediapipe as mp
from model import score

@dataclass
class LiveInferencePipeline:
    model: object = None
    image_size: tuple = (224, 224)  # (width, height) before rotation
    color_mode: str = 'rgb'
    video_source: str = None
    output_path: str = None
    save_video: bool = False
    display_output: bool = True
    fps_display: bool = True
    debug_flag: bool = False
    run_infer: bool = False
    feature_extractor: MasterFeatureExtractor = None

    def __post_init__(self):
        if self.video_source is None:
            raise ValueError("A video source must be specified")

        # After resize+rotate dims
        w, h = self.image_size
        self.processed_width, self.processed_height = h, w

        # Text settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_color = (0, 255, 0)
        self.line_type = 2

    def _load_video_source(self):
        if self.debug_flag:
            print(f"DEBUG: Opening video source: {self.video_source}")
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.debug_flag:
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"DEBUG: Source size: {int(w)}x{int(h)}, FPS: {self.fps}")
        return cap

    def _setup_video_writer(self):
        if not (self.save_video and self.output_path):
            return None
        if self.debug_flag:
            print(f"DEBUG: Writer at {self.output_path}, size=({self.processed_width}x{self.processed_height})")
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(self.output_path, fourcc, self.fps,
                               (self.processed_width, self.processed_height))

    def _preprocess_frame(self, frame):
        if self.debug_flag:
            print(f"DEBUG: Preprocess shape={frame.shape}")
        pf = cv2.resize(frame, self.image_size)
        print(f"DEBUG: Preprocessed shape={pf.shape}")
            
        if self.color_mode == 'rgb':
            pf = cv2.cvtColor(pf, cv2.COLOR_BGR2RGB)
        elif self.color_mode == 'gray':
            pf = cv2.cvtColor(pf, cv2.COLOR_BGR2GRAY)
        pf = cv2.rotate(pf, cv2.ROTATE_90_CLOCKWISE)
        return pf

    def _run_inference(self, pf):
        if self.debug_flag:
            print("DEBUG: Running inference")
        try:
            print("DEBUG: Running infederence")
            feature, left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks = self.feature_extractor.get_feature_for_model(pf, 0)
            var0, label = score(feature)
            return label, left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks
        except Exception as e:
            feature = self.feature_extractor.get_feature_for_model(pf, 0)
            if self.debug_flag:
                print(f"DEBUG: Inference error: {e}\n{traceback.format_exc()}")
                print(feature)
            return "Error"

    def _annotate_frame(self, frame, label, left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks):
        af = frame.copy()
        text = f"{label}"
        cv2.putText(af, text, (10, 30), self.font, self.font_scale,
                    self.font_color, self.line_type)
        def draw_landmarks(landmarks, color=(0, 255, 0)):
            if landmarks is None:
                return
            for lm in landmarks:
                if lm is None:
                    continue
                x, y = int(lm[0]), int(lm[1])
                cv2.circle(af, (x, y), 2, color, -1)

        # Draw all landmarks
        draw_landmarks(left_eye_landmarks, color=(255, 0, 0))
        draw_landmarks(right_eye_landmarks, color=(0, 255, 0))
        draw_landmarks(left_pupil_landmarks, color=(0, 0, 255))
        draw_landmarks(right_pupil_landmarks, color=(255, 255, 0))
        draw_landmarks(mouth_landmarks, color=(255, 0, 255))

        # Draw hand landmarks if available
        if isinstance(hand_landmarks, dict):
            hand_points = [(hand_landmarks[k], hand_landmarks[k.replace("_x", "_y")], hand_landmarks.get(k.replace("_x", "_z"), 0))
                        for k in hand_landmarks if "_x" in k]
            draw_landmarks(hand_points, color=(0, 0, 0))

        return af

    def run(self):
        cap, out = None, None
        try:
            cap = self._load_video_source()
            out = self._setup_video_writer()
            frame_count, fps = 0, 0
            prev_time = time.time()

            if self.debug_flag:
                print("DEBUG: Main loop start")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame (no parallelization)
                processed = self._preprocess_frame(frame)

                if self.color_mode == 'rgb':
                    write_frame = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                elif self.color_mode == 'gray':
                    write_frame = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                else:
                    write_frame = processed

                # inference & annotation
                if self.run_infer:
                    label , left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks= self._run_inference(processed)
                    #print(left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks)
                    disp = self._annotate_frame(write_frame, label,  left_eye_landmarks, right_eye_landmarks, left_pupil_landmarks, right_pupil_landmarks, mouth_landmarks, hand_landmarks)
                else:
                    disp = write_frame

                # save
                if self.save_video and out:
                    out.write(write_frame)

                # display
                if self.display_output:
                    cv2.imshow("Inference Output", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # fps
                frame_count += 1
                now = time.time()
                if now - prev_time >= 1.0:
                    fps = frame_count / (now - prev_time)
                    frame_count, prev_time = 0, now
                    if self.fps_display and self.debug_flag:
                        print(f"DEBUG: FPS={fps:.2f}")

        except Exception as e:
            print(f"Error in pipeline: {e}")
            if self.debug_flag:
                print(traceback.format_exc())
        finally:
            if self.debug_flag:
                print("DEBUG: Cleaning up")
            if cap: cap.release()
            if out: out.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    videos = os.listdir('./')
    # Create a dataframe with video file name and labe;
    for video in videos:
        if video.endswith('.mp4'):
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            extractor = MasterFeatureExtractor(face_mesh, pose, frame_width=540, frame_height=960, model_path='random')
            pipeline = LiveInferencePipeline(
                model=score,
                image_size=(960, 540),
                color_mode='rgb',
                #video_source='http://10.72.240.138:4747/video',
                video_source= '/home/pranavt/edge ai project/video_20250428_153522.mp4',
                output_path='inference_preprocessed.mp4',
                save_video=True,
                display_output=True,
                fps_display=True,
                debug_flag=True,
                run_infer=True,
                feature_extractor = extractor
            )
            pipeline.run()
            break