#!/usr/bin/env python3
"""
Video Frame Extractor - Main Execution Script
Run this file to extract frames from videos.
"""

import os
import argparse
from video_extractor import VideoFrameExtractor

def run_simple_example():
    """Simple example extraction"""
    print("🎬 Video Frame Extractor")
    print("=" * 30)
    
    # Initialize extractor
    extractor = VideoFrameExtractor()
    
    # Video file path
    video_path = "doc_filmpje.mp4"
    
    if os.path.exists(video_path):
        # Extract frames
        success, output_dir = extractor.extract_frames(
            video_path=video_path,
            output_dir="frames_aimen_documentary",
            capture_rate=2  # Extract 1 frame per second
        )
        
        if success:
            print(f"\n✅ Success! Frames saved to '{output_dir}' folder")
        else:
            print("\n❌ Extraction failed")
    else:
        print(f"⚠️  Video file not found: {video_path}")
        print("💡 Place your video file at 'video/test_video.mp4'")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files",
        epilog="""
Examples:
  python run.py                                    # Simple example with default video
  python run.py video.mp4                         # Extract all frames
  python run.py video.mp4 --duration 30           # Extract first 30 seconds
  python run.py video.mp4 --start-time 10 --duration 20  # Extract 20s from 10s mark
  python run.py video.mp4 --capture-rate 2        # Extract 2 frames per second
  python run.py video.mp4 --info                  # Show video information
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('video_path', nargs='?', help='Path to input video file')
    parser.add_argument('--output-dir', default='frames', help='Output directory (default: frames)')
    parser.add_argument('--start-time', type=int, help='Start time in seconds')
    parser.add_argument('--duration', type=int, help='Duration in seconds to extract')
    parser.add_argument('--capture-rate', type=int, help='Frames per second to capture')
    parser.add_argument('--info', action='store_true', help='Show video info only')
    
    args = parser.parse_args()
    
    # If no video path provided, run simple example
    if not args.video_path:
        run_simple_example()
        return
    
    # Initialize extractor
    extractor = VideoFrameExtractor()
    
    if args.info:
        # Show video information
        info = extractor.get_video_info(args.video_path)
        if info:
            print(f"\n📹 Video Information:")
            print(f"   Filename: {info['filename']}")
            print(f"   Resolution: {info['width']}x{info['height']}")
            print(f"   FPS: {info['fps']:.1f}")
            print(f"   Duration: {info['duration_seconds']:.1f} seconds")
            print(f"   Total Frames: {info['frame_count']}")
            print(f"   File Size: {info['size_mb']:.1f} MB")
        else:
            print("❌ Could not read video information")
    else:
        # Extract frames
        success, output_dir = extractor.extract_frames(
            video_path=args.video_path,
            output_dir=args.output_dir,
            start_time=args.start_time,
            duration=args.duration,
            capture_rate=args.capture_rate
        )
        
        if not success:
            exit(1)

if __name__ == "__main__":
    main() 