#!/usr/bin/env python3
"""
Simple Video Frame Extractor
Extract frames from video files using OpenCV with simple naming.

Author: ML4QS Team
Version: 1.0
"""

import os
import cv2
from pathlib import Path
from typing import Tuple


class VideoFrameExtractor:
    """Simple video frame extraction using OpenCV"""
    
    def __init__(self):
        """Initialize the extractor"""
        pass
    
    def extract_frames(self, 
                      video_path: str,
                      output_dir: str = "frames",
                      start_time: int = None,
                      duration: int = None,
                      capture_rate: int = None) -> Tuple[bool, str]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to input video file
            output_dir: Output directory for frames
            start_time: Start time in seconds (optional, default: 0)
            duration: Duration in seconds to extract (optional, default: entire video)
            capture_rate: Frames per second to capture (optional, uses original if None)
            
        Returns:
            Tuple[bool, str]: (Success status, output directory path)
        """
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False, ""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return False, ""
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            
            print(f"📹 Video: {Path(video_path).name}")
            print(f"   FPS: {fps:.1f}, Duration: {video_duration:.1f}s, Total frames: {total_frames}")
            
            # Calculate frame range
            start_frame = int((start_time or 0) * fps)
            
            if duration:
                end_time = (start_time or 0) + duration
                end_frame = int(end_time * fps)
                print(f"   Extracting {duration}s from {start_time or 0}s to {end_time}s")
            else:
                end_frame = total_frames
                if start_time:
                    print(f"   Extracting from {start_time}s to end ({video_duration:.1f}s)")
                else:
                    print(f"   Extracting entire video ({video_duration:.1f}s)")
            
            # Validate range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
            # Calculate frame skip
            if capture_rate:
                frame_skip = max(1, int(fps / capture_rate))
                print(f"   Extracting at {capture_rate} fps (every {frame_skip} frames)")
            else:
                frame_skip = 1
                print(f"   Extracting all frames")
            
            # Extract frames
            frame_count = 0
            current_frame = start_frame
            
            print(f"📁 Saving frames to: {output_dir}")
            
            while current_frame < end_frame:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if ret:
                    # Flip frame 180 degrees
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    
                    # Calculate the timestamp in seconds
                    timestamp_seconds = current_frame / fps

                    # Round to nearest 0.5 seconds
                    timestamp_rounded = round(timestamp_seconds * 2) / 2

                    # Format filename with the timestamp
                    frame_filename = f"frame_{timestamp_rounded:.1f}s.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)

                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                
                current_frame += frame_skip
            
            cap.release()
            
            print(f"✅ Extracted {frame_count} frames successfully!")
            return True, output_dir
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False, ""
    
    def get_video_info(self, video_path: str) -> dict:
        """Get basic video information"""
        if not os.path.exists(video_path):
            return None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            info = {
                'filename': Path(video_path).name,
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                'size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
            
            cap.release()
            return info
            
        except Exception:
            return None 