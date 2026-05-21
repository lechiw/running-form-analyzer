"""
tests/conftest.py - Shared test fixtures for Running Form Analyzer
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from pose_extractor import PoseLandmarks, PoseSequence
from metrics import L


@pytest.fixture
def synthetic_sequence():
    """Generate 10s of synthetic running data at 30fps."""
    n_frames = 300
    seq = PoseSequence(fps=30, total_frames=n_frames, video_path='synthetic')
    rng = np.random.RandomState(42)

    for t in range(n_frames):
        phase = t / 30.0 * 2 * np.pi * 3.0  # ~180 spm
        lm = np.zeros((33, 3))
        vis = np.ones(33) * 0.92

        hip_y = 500 + 12 * np.sin(phase)
        hip_x = 640 + 5 * np.cos(phase * 0.5)

        lm[L['L_HIP']] = [hip_x - 30, hip_y + 2, 0]
        lm[L['R_HIP']] = [hip_x + 30, hip_y - 2, 0]

        # Shoulders with ~5° forward lean
        lean_rad = 0.087
        sy = 160 * np.cos(lean_rad)
        sx = 160 * np.sin(lean_rad)
        lm[L['L_SHOULDER']] = [hip_x - 40 + sx, hip_y - sy, 0]
        lm[L['R_SHOULDER']] = [hip_x + 40 + sx, hip_y - sy, 0]

        # Arms
        arm_swing = 30 * np.sin(phase)
        lm[L['L_ELBOW']] = [hip_x - 70 - arm_swing, hip_y - 120, 0]
        lm[L['R_ELBOW']] = [hip_x + 70 + arm_swing, hip_y - 120, 0]
        lm[L['L_WRIST']] = [hip_x - 100 - arm_swing*1.3, hip_y - 70, 0]
        lm[L['R_WRIST']] = [hip_x + 100 + arm_swing*1.3, hip_y - 70, 0]

        # Legs
        leg_p = phase
        lm[L['L_KNEE']] = [hip_x - 40 + 40*np.cos(leg_p), hip_y + 90 + 30*np.sin(leg_p), 0]
        lm[L['R_KNEE']] = [hip_x + 40 + 40*np.cos(leg_p+np.pi), hip_y + 90 + 30*np.sin(leg_p+np.pi), 0]
        lm[L['L_ANKLE']] = [hip_x - 50 + 60*np.cos(leg_p), hip_y + 180 + 40*np.sin(leg_p), 0]
        lm[L['R_ANKLE']] = [hip_x + 50 + 60*np.cos(leg_p+np.pi), hip_y + 180 + 40*np.sin(leg_p+np.pi), 0]
        lm[L['L_HEEL']] = [hip_x - 50 + 60*np.cos(leg_p), hip_y + 195 + 40*np.sin(leg_p), 0]
        lm[L['R_HEEL']] = [hip_x + 50 + 60*np.cos(leg_p+np.pi), hip_y + 195 + 40*np.sin(leg_p+np.pi), 0]
        lm[L['L_FOOT']] = [hip_x - 60 + 60*np.cos(leg_p), hip_y + 195 + 40*np.sin(leg_p), 0]
        lm[L['R_FOOT']] = [hip_x + 60 + 60*np.cos(leg_p+np.pi), hip_y + 195 + 40*np.sin(leg_p+np.pi), 0]
        lm[0] = [hip_x + sx, hip_y - sy - 50, 0]

        seq.landmarks_seq.append(PoseLandmarks(
            frame_idx=t, landmarks=lm, visibility=vis, timestamp_ms=t/30*1000
        ))

    return seq


@pytest.fixture
def empty_sequence():
    """Empty pose sequence."""
    return PoseSequence(fps=30, total_frames=0)


@pytest.fixture
def low_visibility_sequence():
    """Sequence with very low visibility (should be filtered out)."""
    seq = PoseSequence(fps=30, total_frames=30, video_path='low_vis')
    for t in range(30):
        lm = np.zeros((33, 3))
        vis = np.ones(33) * 0.1  # Very low visibility
        seq.landmarks_seq.append(PoseLandmarks(
            frame_idx=t, landmarks=lm, visibility=vis, timestamp_ms=t/30*1000
        ))
    return seq
