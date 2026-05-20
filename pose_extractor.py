"""
pose_extractor.py - MediaPipe Pose Extraction for Running Form Analysis
Uses the MediaPipe Solutions API (Pose) for reliable CPU inference.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PoseLandmarks:
    """Stores full-body landmark data for a single frame."""
    frame_idx: int
    landmarks: np.ndarray  # shape (33, 3) -> (x, y, z) in pixel coords
    visibility: np.ndarray  # shape (33,) confidence per landmark
    timestamp_ms: float


@dataclass
class PoseSequence:
    """Stores pose data across all frames of a video."""
    landmarks_seq: List[PoseLandmarks] = field(default_factory=list)
    fps: float = 0.0
    total_frames: int = 0
    video_path: str = ""

    def to_dict(self) -> dict:
        """Convert to serializable dict for LLM analysis."""
        if not self.landmarks_seq:
            return {}
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": len(self.landmarks_seq) / self.fps if self.fps > 0 else 0,
            "landmarks": [
                {
                    "frame": lm.frame_idx,
                    "data": lm.landmarks.tolist(),
                    "visibility": lm.visibility.tolist(),
                }
                for lm in self.landmarks_seq
            ],
        }


class PoseExtractor:
    """Wrapper around MediaPipe Pose for running form analysis."""

    # Key landmarks for running analysis
    KEY_LANDMARKS = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_heel": 29, "right_heel": 30,
        "left_foot": 31, "right_foot": 32,
    }

    # All landmark names
    LANDMARK_NAMES = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist",
        17: "left_pinky", 18: "right_pinky",
        19: "left_index", 20: "right_index",
        21: "left_thumb", 22: "right_thumb",
        23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle",
        29: "left_heel", 30: "right_heel",
        31: "left_foot_index", 32: "right_foot_index",
    }

    # Landmark connections for drawing
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            model_complexity: 0=lite, 1=full, 2=heavy (CPU best at 1)
            min_detection_confidence: minimum detection confidence
            min_tracking_confidence: minimum tracking confidence
        """
        self.model_complexity = model_complexity
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Use model_complexity=1 (full) if 2 (heavy) model not available
        if model_complexity == 2:
            import os
            heavy_path = os.path.join(os.path.dirname(mp.solutions.pose.__file__),
                                      'pose_landmark_heavy.tflite')
            if not os.path.exists(heavy_path):
                print("   Heavy model not cached, falling back to full (model_complexity=1)")
                model_complexity = 1

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_from_video(self, video_path: str, max_frames: Optional[int] = None,
                           stride: int = 1) -> PoseSequence:
        """
        Extract pose landmarks from a video.

        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process
            stride: Process every N frames

        Returns:
            PoseSequence with extracted landmarks
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 Video: {video_path}")
        print(f"   Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
        print(f"   Processing every {stride} frame(s)...")

        seq = PoseSequence(fps=fps, total_frames=total_frames, video_path=video_path)
        frame_idx = 0
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            if max_frames and processed >= max_frames:
                break

            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                landmarks = np.zeros((33, 3))
                visibility = np.zeros(33)

                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks[i] = [lm.x * w, lm.y * h, lm.z * w]
                    visibility[i] = lm.visibility

                seq.landmarks_seq.append(PoseLandmarks(
                    frame_idx=frame_idx,
                    landmarks=landmarks,
                    visibility=visibility,
                    timestamp_ms=(frame_idx / fps) * 1000 if fps > 0 else 0,
                ))

            processed += 1
            frame_idx += 1

            if processed % 200 == 0:
                print(f"   Processed {processed} frames...")

        cap.release()
        self.pose.close()

        print(f"✅ Done: {len(seq.landmarks_seq)}/{processed} frames with valid pose data")
        return seq

    def extract_from_image(self, image_path: str) -> Optional[PoseLandmarks]:
        """Extract pose from a single image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = img.shape[:2]
        landmarks = np.zeros((33, 3))
        visibility = np.zeros(33)

        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x * w, lm.y * h, lm.z * w]
            visibility[i] = lm.visibility

        return PoseLandmarks(frame_idx=0, landmarks=landmarks, visibility=visibility, timestamp_ms=0)

    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks_data: PoseLandmarks,
                       draw_skeleton: bool = True, draw_labels: bool = False,
                       highlight_joints: List[int] = None) -> np.ndarray:
        """Draw pose landmarks on a frame for visualization."""
        vis_frame = frame.copy()

        if draw_skeleton:
            for connection in PoseExtractor.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if (start_idx < len(landmarks_data.landmarks) and
                    end_idx < len(landmarks_data.landmarks)):
                    start = landmarks_data.landmarks[start_idx]
                    end = landmarks_data.landmarks[end_idx]
                    start_vis = landmarks_data.visibility[start_idx]
                    end_vis = landmarks_data.visibility[end_idx]

                    if start_vis > 0.5 and end_vis > 0.5:
                        pt1 = (int(start[0]), int(start[1]))
                        pt2 = (int(end[0]), int(end[1]))
                        cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2)

            for i, (lm, vis) in enumerate(zip(landmarks_data.landmarks,
                                              landmarks_data.visibility)):
                if vis > 0.5:
                    x, y = int(lm[0]), int(lm[1])
                    color = (0, 0, 255)
                    if highlight_joints and i in highlight_joints:
                        color = (0, 255, 255)
                        cv2.circle(vis_frame, (x, y), 8, color, -1)
                    else:
                        cv2.circle(vis_frame, (x, y), 4, color, -1)

                    if draw_labels and i in PoseExtractor.KEY_LANDMARKS.values():
                        name = [k for k, v in PoseExtractor.KEY_LANDMARKS.items()
                                if v == i]
                        if name:
                            cv2.putText(vis_frame, name[0], (x + 8, y - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 1)

        return vis_frame
