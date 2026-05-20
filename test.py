#!/usr/bin/env python3
"""
test.py - Quick smoke test for Running Form Analyzer
Tests the full pipeline with synthetic data to verify everything works.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pose_extractor import PoseExtractor, PoseLandmarks, PoseSequence
from metrics import RunningMetricsCalculator, L as LM
from visualizer import RunningFormVisualizer
from analyzer import AIRunningCoach


def generate_synthetic_running_data(fps=30, duration_sec=10):
    """Generate synthetic running pose sequence for testing."""
    n_frames = int(fps * duration_sec)
    h, w = 720, 1280
    rng = np.random.RandomState(42)

    seq = PoseSequence(fps=fps, total_frames=n_frames, video_path='synthetic')

    for t in range(n_frames):
        phase = t / fps * 2 * np.pi * 3.0  # ~180 steps/min (realistic cadence)
        lm = np.zeros((33, 3))
        vis = np.ones(33) * 0.92

        # Hip oscillation
        hip_y = 500 + 12 * np.sin(phase)
        hip_x = w / 2 + 5 * np.cos(phase * 0.5)

        lm[LM['L_HIP']] = [hip_x - 30, hip_y, 0]
        lm[LM['R_HIP']] = [hip_x + 30, hip_y, 0]

        # Shoulder (slight counter-rotation)
        lm[LM['L_SHOULDER']] = [hip_x - 40, hip_y - 160, 0]
        lm[LM['R_SHOULDER']] = [hip_x + 40, hip_y - 160, 0]

        # Trunk lean: ~5° forward
        lean = 0.087  # 5 degrees in radians
        shoulder_y_offset = 160 * np.cos(lean)
        shoulder_x_offset = 160 * np.sin(lean)
        lm[LM['L_SHOULDER']] = [hip_x - 40 + shoulder_x_offset, hip_y - shoulder_y_offset, 0]
        lm[LM['R_SHOULDER']] = [hip_x + 40 + shoulder_x_offset, hip_y - shoulder_y_offset, 0]

        # Arms swinging (anti-phase to legs)
        arm_swing = 30 * np.sin(phase)
        lm[LM['L_ELBOW']] = [hip_x - 70 - arm_swing, hip_y - 120, 0]
        lm[LM['R_ELBOW']] = [hip_x + 70 + arm_swing, hip_y - 120, 0]
        lm[LM['L_WRIST']] = [hip_x - 100 - arm_swing*1.3, hip_y - 70, 0]
        lm[LM['R_WRIST']] = [hip_x + 100 + arm_swing*1.3, hip_y - 70, 0]

        # Legs in running gait
        leg_phase = phase
        lm[LM['L_KNEE']] = [hip_x - 40 + 40*np.cos(leg_phase), hip_y + 90 + 30*np.sin(leg_phase), 0]
        lm[LM['R_KNEE']] = [hip_x + 40 + 40*np.cos(leg_phase+np.pi), hip_y + 90 + 30*np.sin(leg_phase+np.pi), 0]
        lm[LM['L_ANKLE']] = [hip_x - 50 + 60*np.cos(leg_phase), hip_y + 180 + 40*np.sin(leg_phase), 0]
        lm[LM['R_ANKLE']] = [hip_x + 50 + 60*np.cos(leg_phase+np.pi), hip_y + 180 + 40*np.sin(leg_phase+np.pi), 0]
        lm[LM['L_HEEL']] = [hip_x - 50 + 60*np.cos(leg_phase), hip_y + 195 + 40*np.sin(leg_phase), 0]
        lm[LM['R_HEEL']] = [hip_x + 50 + 60*np.cos(leg_phase+np.pi), hip_y + 195 + 40*np.sin(leg_phase+np.pi), 0]
        lm[LM['L_FOOT']] = [hip_x - 60 + 60*np.cos(leg_phase), hip_y + 195 + 40*np.sin(leg_phase), 0]
        lm[LM['R_FOOT']] = [hip_x + 60 + 60*np.cos(leg_phase+np.pi), hip_y + 195 + 40*np.sin(leg_phase+np.pi), 0]

        # Head (nose)
        lm[0] = [hip_x + shoulder_x_offset, hip_y - shoulder_y_offset - 50, 0]

        seq.landmarks_seq.append(PoseLandmarks(
            frame_idx=t, landmarks=lm, visibility=vis, timestamp_ms=t/fps*1000
        ))

    return seq


def test():
    print("=" * 60)
    print("🏃 Running Form Analyzer - Smoke Test")
    print("=" * 60)

    # Step 1: Extract (synthetic data directly)
    print("\n📐 Step 1: Generating synthetic running data...")
    seq = generate_synthetic_running_data(fps=30, duration_sec=10)
    print(f"   ✅ Generated {len(seq.landmarks_seq)} frames at {seq.fps}fps")

    # Step 2: Compute metrics
    print("\n📊 Step 2: Computing running metrics...")
    calc = RunningMetricsCalculator()
    metrics = calc.compute(seq)
    scoring = calc.get_scoring(metrics)

    summary = metrics.summary()
    for k, v in summary.items():
        print(f"   {k}: {v}")

    print(f"\n   🏆 Overall Score: {scoring['overall_score']}/100")
    for k, v in scoring.get('details', {}).items():
        print(f"      {k}: {v}")

    # Step 3: Generate report
    print("\n📝 Step 3: Generating analysis report...")
    coach = AIRunningCoach()
    report = coach.generate_report(metrics, scoring)
    print()
    print(report)

    # Step 4: Visualizer module works
    print("\n🎬 Step 4: Visualizer import check...")
    v = RunningFormVisualizer()
    print(f"   ✅ RunningFormVisualizer loaded (panel width: {v.width})")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print()
    print("🚀 Ready to analyze real videos!")
    print("   Run: python run.py /path/to/your/running_video.mp4 --render")
    print("=" * 60)


if __name__ == "__main__":
    test()
