# Motion Tracker

A real-time motion tracking application using OpenCV that tracks user-selected points and saves motion data to CSV files.

## Features
- Click to select tracking points on live video feed
- Real-time motion tracking using Lucas-Kanade Optical Flow
- Visual motion trails with coordinate display
- Export tracked motion data to timestamped CSV files
- Multiple keyboard controls for interaction

## Setup
1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python motiondetection.py
```

## Controls
- **Left Click**: Select a point to track
- **S**: Save tracked motion data to CSV
- **C**: Clear motion trail
- **R**: Reset tracking
- **ESC**: Exit application

## Data Export
- Motion data is automatically saved to `motion_data/` directory
- CSV files include timestamps, elapsed time, and x/y coordinates
- Files are named with timestamp: `tracked_motion_YYYYMMDD_HHMMSS.csv`
- Option to save data on exit if tracking data exists

## Output Format
CSV columns: `timestamp`, `time_elapsed_seconds`, `x_coordinate`, `y_coordinate`

## Code Explanation
- Captures video from the webcam using OpenCV.
- Converts frames to grayscale for processing.
- Uses `cv2.calcOpticalFlowPyrLK` for tracking movement.
- Updates and draws the motion trail on the frame.

## Output
https://github.com/user-attachments/assets/72b11f6d-3714-480f-ae26-4bf882e55f6b

