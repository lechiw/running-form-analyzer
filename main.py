"""
main.py - Running Form Analyzer CLI
Entry point: analyze a running video and get pose data, metrics, and report.

Usage:
    python main.py path/to/running_video.mp4 [--render] [--output-dir ./output]
"""

import argparse
import os
import sys
import json
from typing import Optional

from pose_extractor import PoseExtractor, PoseSequence
from metrics import RunningMetricsCalculator, RunningMetrics
from visualizer import RunningFormVisualizer
from analyzer import AIRunningCoach


def analyze_video(video_path: str,
                  render: bool = False,
                  output_dir: str = "./output",
                  max_frames: Optional[int] = None,
                  stride: int = 2) -> dict:
    """
    Full pipeline: extract pose, compute metrics, generate report.

    Args:
        video_path: Path to input video
        render: Whether to render annotated video
        output_dir: Output directory for results
        max_frames: Max frames to process (None = all)
        stride: Process every N frames

    Returns:
        Dict with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    print("=" * 60)
    print("🏃 Running Form Analyzer")
    print("=" * 60)

    # Step 1: Extract pose landmarks
    print("\n📐 Step 1/4: Extracting pose landmarks...")
    extractor = PoseExtractor(model_complexity=2)
    seq = extractor.extract_from_video(video_path, max_frames=max_frames, stride=stride)

    if len(seq.landmarks_seq) == 0:
        print("❌ No pose data extracted. Is there a person visible in the video?")
        return {"error": "No pose detected"}

    # Save raw landmarks as JSON
    landmarks_path = os.path.join(output_dir, f"{base_name}_landmarks.json")
    with open(landmarks_path, "w") as f:
        json.dump(seq.to_dict(), f, indent=2)
    print(f"   Raw landmarks saved to: {landmarks_path}")

    # Step 2: Compute running metrics
    print("\n📊 Step 2/4: Computing running metrics...")
    calculator = RunningMetricsCalculator()
    metrics = calculator.compute(seq)
    print(f"   Duration: {metrics.duration_sec:.1f}s")
    print(f"   Cadence: {metrics.cadence_avg:.0f} spm" if metrics.cadence_avg else "   Cadence: N/A")
    print(f"   Trunk lean: {metrics.trunk_lean_avg:.1f}°" if metrics.trunk_lean_avg else "   Trunk lean: N/A")

    # Score
    scoring = calculator.get_scoring(metrics)
    print(f"\n   🏆 Running Form Score: {scoring['overall_score']}/100"
          if scoring['overall_score'] else "   🏆 Score: N/A")

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "summary": metrics.summary(),
            "scoring": scoring,
        }, f, indent=2)
    print(f"   Metrics saved to: {metrics_path}")

    # Step 3: Render annotated video (optional)
    if render:
        print("\n🎬 Step 3/4: Rendering annotated video...")
        output_video = os.path.join(output_dir, f"{base_name}_analysis.mp4")
        visualizer = RunningFormVisualizer()
        visualizer.render_video(video_path, output_video, seq, metrics)
    else:
        output_video = None

    # Step 4: Generate report
    print("\n📝 Step 4/4: Generating analysis report...")
    coach = AIRunningCoach()  # No LLM = template report
    report = coach.generate_report(metrics, scoring)

    report_path = os.path.join(output_dir, f"{base_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"   Report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print(f"   Results directory: {output_dir}")
    print("=" * 60)

    return {
        "landmarks_path": landmarks_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
        "video_output": output_video,
        "metrics": metrics.summary(),
        "score": scoring,
    }


def main():
    parser = argparse.ArgumentParser(description="🏃 Running Form Analyzer")
    parser.add_argument("video", help="Path to running video file")
    parser.add_argument("--render", action="store_true",
                        help="Render annotated video with skeleton")
    parser.add_argument("--output-dir", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (default: all)")
    parser.add_argument("--stride", type=int, default=2,
                        help="Process every N frames (default: 2)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)

    results = analyze_video(
        video_path=args.video,
        render=args.render,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        stride=args.stride,
    )

    # Print report to console
    if "error" not in results:
        report_path = results.get("report_path")
        if report_path and os.path.exists(report_path):
            print("\n" + "=" * 60)
            print("📋 ANALYSIS REPORT")
            print("=" * 60)
            with open(report_path) as f:
                print(f.read())


if __name__ == "__main__":
    main()
