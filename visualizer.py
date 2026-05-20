"""
visualizer.py - Running Form Visualizer
Creates annotated video with skeleton overlay and real-time metrics panel.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from pose_extractor import PoseExtractor, PoseLandmarks, PoseSequence
from metrics import RunningMetrics, FrameMetrics, RunningMetricsCalculator, L


class RunningFormVisualizer:
    """Draws skeleton overlay + metrics panel on running video."""

    # Colors (BGR)
    COLOR_SKELETON = (100, 200, 100)  # green
    COLOR_JOINT = (0, 100, 255)  # orange
    COLOR_HIGHLIGHT = (0, 255, 255)  # yellow
    COLOR_TEXT = (255, 255, 255)  # white
    COLOR_BG = (40, 40, 40)  # dark gray
    COLOR_BAD = (50, 50, 200)  # red
    COLOR_GOOD = (50, 200, 50)  # green

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

    def create_annotated_frame(self, frame: np.ndarray,
                                landmarks_data: Optional[PoseLandmarks],
                                frame_metrics: Optional[FrameMetrics] = None,
                                overall_metrics: Optional[RunningMetrics] = None,
                                show_skeleton: bool = True,
                                show_angles: bool = True) -> np.ndarray:
        """Create an annotated frame with overlay information."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        if landmarks_data is not None and show_skeleton:
            # Draw skeleton
            connections = POSE_CONNECTIONS
            for conn in connections:
                s, e = conn
                if (s < len(landmarks_data.landmarks) and
                    e < len(landmarks_data.landmarks) and
                    landmarks_data.visibility[s] > 0.5 and
                    landmarks_data.visibility[e] > 0.5):
                    pt1 = (int(landmarks_data.landmarks[s][0]),
                           int(landmarks_data.landmarks[s][1]))
                    pt2 = (int(landmarks_data.landmarks[e][0]),
                           int(landmarks_data.landmarks[e][1]))
                    cv2.line(vis, pt1, pt2, self.COLOR_SKELETON, 2)

            # Draw joints
            for i, (lm, v) in enumerate(zip(landmarks_data.landmarks,
                                            landmarks_data.visibility)):
                if v > 0.5:
                    x, y = int(lm[0]), int(lm[1])
                    cv2.circle(vis, (x, y), 5, self.COLOR_JOINT, -1)

            # Draw angle annotations
            if show_angles and frame_metrics is not None:
                self._draw_angle_annotations(vis, landmarks_data, frame_metrics)

        # Draw metrics panel (right side overlay)
        vis = self._draw_metrics_panel(vis, frame_metrics, overall_metrics)

        return vis

    def _draw_angle_annotations(self, frame: np.ndarray,
                                 landmarks: PoseLandmarks,
                                 metrics: FrameMetrics):
        """Draw angle text near relevant joints."""
        lm = landmarks.landmarks
        h, w = frame.shape[:2]

        # Trunk lean
        if metrics.trunk_lean_angle is not None:
            mid_hip = _midpoint(lm[L["L_HIP"]], lm[L["R_HIP"]])
            x, y = int(mid_hip[0]), int(mid_hip[1])
            cv2.putText(frame, f"Lean: {metrics.trunk_lean_angle:.1f}°",
                        (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, self.COLOR_HIGHLIGHT, 2)

        # Left elbow
        if metrics.left_elbow_angle is not None:
            x, y = int(lm[L["L_ELBOW"]][0]), int(lm[L["L_ELBOW"]][1])
            cv2.putText(frame, f"L: {metrics.left_elbow_angle:.0f}°",
                        (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, self.COLOR_HIGHLIGHT, 1)

        # Right elbow
        if metrics.right_elbow_angle is not None:
            x, y = int(lm[L["R_ELBOW"]][0]), int(lm[L["R_ELBOW"]][1])
            cv2.putText(frame, f"R: {metrics.right_elbow_angle:.0f}°",
                        (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, self.COLOR_HIGHLIGHT, 1)

    def _draw_metrics_panel(self, frame: np.ndarray,
                             frame_metrics: Optional[FrameMetrics],
                             overall: Optional[RunningMetrics]) -> np.ndarray:
        """Draw a semi-transparent metrics panel on the right side."""
        h, w = frame.shape[:2]
        panel_w = 320
        panel_x = w - panel_w - 10
        panel_y = 10

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + 280),
                      self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "🏃 Running Form Analysis",
                    (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_TEXT, 2)

        y_offset = panel_y + 50
        line_h = 22

        if overall:
            metrics = [
                ("Cadence", f"{overall.cadence_avg:.0f} spm" if overall.cadence_avg else "N/A",
                 "GOOD" if overall.cadence_avg and 165 <= overall.cadence_avg <= 185 else "CHECK"),
                ("Trunk Lean", f"{overall.trunk_lean_avg:.1f}°" if overall.trunk_lean_avg else "N/A",
                 "GOOD" if overall.trunk_lean_avg and 4 <= overall.trunk_lean_avg <= 12 else "CHECK"),
                ("Arm Symmetry", f"{overall.arm_symmetry_avg:.0f}/100" if overall.arm_symmetry_avg else "N/A",
                 "GOOD" if overall.arm_symmetry_avg and overall.arm_symmetry_avg >= 80 else "CHECK"),
                ("Vert Osc", f"{overall.vertical_oscillation:.1f}cm" if overall.vertical_oscillation else "N/A",
                 "GOOD" if overall.vertical_oscillation and overall.vertical_oscillation < 10 else "CHECK"),
                ("Foot Strike", f"{overall.avg_foot_strike_distance:.1f}cm" if overall.avg_foot_strike_distance else "N/A",
                 "GOOD" if overall.avg_foot_strike_distance and overall.avg_foot_strike_distance < 15 else "CHECK"),
                ("Strike Type", overall.foot_strike_type_dominant or "N/A", "INFO"),
            ]

            for label, value, status in metrics:
                color = self.COLOR_GOOD if status == "GOOD" else self.COLOR_BAD if status == "CHECK" else self.COLOR_TEXT
                cv2.putText(frame, f"{label}:", (panel_x + 15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT, 1)
                cv2.putText(frame, value, (panel_x + 160, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                y_offset += line_h

        # Frame-level realtime metrics (if available)
        if frame_metrics:
            y_offset += 5
            cv2.putText(frame, "--- Live ---", (panel_x + 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT, 1)
            y_offset += line_h

            live_metrics = []
            if frame_metrics.trunk_lean_angle is not None:
                live_metrics.append(f"Lean: {frame_metrics.trunk_lean_angle:.1f}°")
            if frame_metrics.left_elbow_angle is not None:
                live_metrics.append(f"L Elbow: {frame_metrics.left_elbow_angle:.0f}°")
            if frame_metrics.right_elbow_angle is not None:
                live_metrics.append(f"R Elbow: {frame_metrics.right_elbow_angle:.0f}°")
            if frame_metrics.arm_symmetry_score is not None:
                live_metrics.append(f"Sym: {frame_metrics.arm_symmetry_score:.0f}%")
            if frame_metrics.foot_strike_distance is not None:
                live_metrics.append(f"Strike: {frame_metrics.foot_strike_distance:.1f}cm")

            for metric in live_metrics:
                cv2.putText(frame, metric, (panel_x + 15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_HIGHLIGHT, 1)
                y_offset += line_h - 2

        return frame

    def render_video(self, video_path: str, output_path: str,
                      seq: PoseSequence, metrics: RunningMetrics) -> str:
        """
        Render annotated video with skeleton overlay and metrics panel.

        Args:
            video_path: Input video path
            output_path: Output video path (e.g., output.mp4)
            seq: Pose sequence data (aligned frame-by-frame)
            metrics: Computed running metrics

        Returns:
            Path to output video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure output dir exists
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Try H.264 for browser compatibility; fall back to mp4v
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Build a frame-to-metrics lookup
        frame_metrics_map = {}
        for fm in metrics.frame_metrics:
            frame_metrics_map[fm.frame_idx] = fm

        # Build a frame-to-landmarks lookup
        landmarks_map = {}
        for pl in seq.landmarks_seq:
            landmarks_map[pl.frame_idx] = pl

        print(f"  Rendering {seq.total_frames} frames...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            lm = landmarks_map.get(frame_idx)
            fm = frame_metrics_map.get(frame_idx)

            annotated = self.create_annotated_frame(
                frame, lm, fm, metrics
            )
            out.write(annotated)

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"  Rendered {frame_idx}/{seq.total_frames}...")

        cap.release()
        out.release()
        print(f"✅ Annotated video saved to: {output_path}")

        return output_path


def _midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Midpoint of two points."""
    return (p1 + p2) / 2


# Landmark connections (hardcoded to avoid MediaPipe import issues)
# From MediaPipe Pose:
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (23, 24), (27, 28)
])
