#!/usr/bin/env python3
"""
run.py - Standalone entry point for Running Form Analyzer

Usage:
    python run.py path/to/running_video.mp4 --render
    python run.py samples/test_run.mp4 --output-dir ./results
"""

import sys
import os

# Ensure the project dir is on sys.path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from main import main

if __name__ == "__main__":
    main()
