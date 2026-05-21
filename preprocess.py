"""
preprocess.py - Video Preprocessing for Running Form Analysis
Handles vertical/portrait videos by auto-cropping to landscape.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# Default target width for preprocessed output
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # ~1.78 (16:9)


def is_portrait_video(video_path: str) -> bool:
    """
    Quick check: is this video in portrait/vertical orientation?
    Returns True if height > width (9:16 or similar).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h > w


def detect_runner_roi(video_path: str, sample_frames: int = 30,
                      margin_ratio: float = 0.15) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the runner's region of interest (ROI) across sampled frames.
    
    Uses simple frame-differencing and brightness analysis to find the
    vertical bounds of the runner, then adds margins.
    
    Returns (x, y, w, h) ROI in original frame coordinates, or None if
    detection fails. The ROI is designed to be cropped and resized to
    produce a wider (16:9) output frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // sample_frames)
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Collect vertical pixel activity across sampled frames
    y_activations = []
    
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find the runner's silhouette
        edges = cv2.Canny(gray, 50, 150)
        
        # Sum edge pixels horizontally (column-wise)
        # This gives us a profile of activity by row
        col_activity = np.sum(edges, axis=1)
        y_activations.append(col_activity)
    
    cap.release()
    
    if not y_activations:
        return None
    
    # Average activity across frames
    avg_activity = np.mean(y_activations, axis=0)
    
    # Find rows with significant activity (above 10% of max)
    threshold = np.max(avg_activity) * 0.05
    active_rows = np.where(avg_activity > threshold)[0]
    
    if len(active_rows) < 10:
        return None
    
    y_min = int(active_rows[0])
    y_max = int(active_rows[-1])
    
    # Add margins
    margin = int((y_max - y_min) * margin_ratio)
    y_min = max(0, y_min - margin)
    y_max = min(frame_h - 1, y_max + margin)
    
    crop_height = y_max - y_min
    
    # Calculate crop width to produce a 16:9 output
    # If we crop (y_min, y_max), we want width such that width/crop_height ≈ 16/9
    target_crop_width = int(crop_height * ASPECT_RATIO)
    
    # Center the crop horizontally
    if target_crop_width >= frame_w:
        # Already wide enough, use full width
        x = 0
        crop_w = frame_w
    else:
        # Crop horizontally to match 16:9
        x = max(0, (frame_w - target_crop_width) // 2)
        crop_w = min(frame_w - x, target_crop_width)
    
    return (x, int(y_min), crop_w, crop_height)


def preprocess_video(input_path: str, output_path: Optional[str] = None,
                     roi: Optional[Tuple[int, int, int, int]] = None,
                     target_width: int = TARGET_WIDTH,
                     target_height: int = TARGET_HEIGHT) -> Optional[str]:
    """
    Preprocess a video (especially portrait) for better analysis.
    
    Crops the frame to the detected ROI and resizes to a standard
    landscape resolution. This helps keep the runner's feet in frame
    even for vertical videos.
    
    Args:
        input_path: Source video path
        output_path: Output path (auto-generated if None)
        roi: Pre-computed ROI (x, y, w, h). Auto-detected if None.
        target_width: Output video width
        target_height: Output video height
    
    Returns:
        Path to preprocessed video, or None if processing failed
    """
    if not is_portrait_video(input_path):
        return None  # Not portrait, no preprocessing needed
    
    # Auto-generate output path
    if output_path is None:
        inp = Path(input_path)
        output_path = str(inp.parent / f"{inp.stem}_preprocessed{inp.suffix}")
    
    # Auto-detect ROI if not provided
    if roi is None:
        roi = detect_runner_roi(input_path)
    if roi is None:
        return None  # Detection failed, return None
    
    x, y, w, h = roi
    if w < 100 or h < 100:
        return None
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Try H.264 for browser compat; fall back to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    if not out.isOpened():
        cap.release()
        return None
    
    print(f"  Preprocessing portrait video: {input_path}")
    print(f"    Original: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"    ROI: ({x},{y},{w},{h})")
    print(f"    Output: {target_width}x{target_height}")
    
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop to ROI
        cropped = frame[y:y+h, x:x+w]
        
        # Resize to target
        resized = cv2.resize(cropped, (target_width, target_height),
                             interpolation=cv2.INTER_LINEAR)
        
        out.write(resized)
    
    cap.release()
    out.release()
    
    print(f"  Preprocessing complete: {output_path}")
    return output_path


def describe_orientation(video_path: str) -> str:
    """Return a human-readable description of video orientation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "无法打开视频"
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    ratio = w / h
    
    if ratio > 1.5:
        return "横屏视频"
    elif ratio < 0.75:
        return f"竖屏视频 ({w}x{h})"
    else:
        return f"接近正方形视频 ({w}x{h})"
