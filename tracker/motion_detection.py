import cv2
import numpy as np
import csv
import datetime
import os

cap = cv2.VideoCapture(0)

# Read first frame and convert to grayscale
ret, frame = cap.read()
if not ret:
    print("Error: Couldn't access webcam.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# List to store tracked points with timestamps
tracked_points = []
start_time = datetime.datetime.now()

def save_tracked_points():
    """Save tracked points to a CSV file"""
    if not tracked_points:
        print("No tracked points to save!")
        return
    
    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tracked_motion_{timestamp}.csv"
    
    # Create directory if it doesn't exist
    os.makedirs("motion_data", exist_ok=True)
    filepath = os.path.join("motion_data", filename)
    
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['timestamp', 'time_elapsed_seconds', 'x_coordinate', 'y_coordinate'])
            
            # Write data
            for point_data in tracked_points:
                writer.writerow(point_data)
        
        print(f"Tracked motion saved to: {filepath}")
        print(f"Total points saved: {len(tracked_points)}")
    except Exception as e:
        print(f"Error saving file: {e}")

def select_point(event, x, y, flags, param):
    global point_selected, point, old_points, tracked_points, start_time
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)
        # Reset tracking data for new point
        tracked_points = []
        start_time = datetime.datetime.now()
        print(f"New point selected at ({x}, {y}). Tracking started.")

cv2.namedWindow("Motion Tracker")
cv2.setMouseCallback("Motion Tracker", select_point)

point_selected = False
point = (0, 0)
old_points = np.array([[]])
mask = np.zeros_like(frame) #develop a empty image to track motion trail

print("Instructions:")
print("- Left click to select a point to track")
print("- Press 's' to save tracked motion data")
print("- Press 'c' to clear motion trail")
print("- Press 'r' to reset tracking")
print("- Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        
        if new_points is not None and len(new_points) > 0:
            x, y = new_points.ravel()
            new_point = (int(x), int(y))

            # Store the tracked point with timestamp
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()
            tracked_points.append([
                current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # timestamp with milliseconds
                round(elapsed_time, 3),  # elapsed time in seconds
                int(x),  # x coordinate
                int(y)   # y coordinate
            ])

            mask = cv2.line(mask, point, new_point, (0, 255, 0), 2)
            frame = cv2.circle(frame, new_point, 5, (0, 255, 0), -1)

            # Update old points and frame and Update tracked point
            old_gray = gray_frame.copy()
            old_points = new_points.reshape(-1, 1, 2)
            point = new_point

        # Merge frame with mask to show motion trails
        frame = cv2.add(frame, mask)

    # Display tracking info
    info_text = f"Tracked points: {len(tracked_points)}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if point_selected:
        coord_text = f"Current: ({point[0]}, {point[1]})"
        cv2.putText(frame, coord_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Motion Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('s'):  # Save tracked points
        save_tracked_points()
    elif key == ord('c'):  # Clear motion trail
        mask = np.zeros_like(frame)
        print("Motion trail cleared.")
    elif key == ord('r'):  # Reset tracking
        point_selected = False
        tracked_points = []
        mask = np.zeros_like(frame)
        print("Tracking reset.")

cap.release()
cv2.destroyAllWindows()

# Auto-save on exit if there are tracked points
if tracked_points:
    print(f"\nExiting... Found {len(tracked_points)} tracked points.")
    save_choice = input("Save tracked motion data? (y/n): ").lower().strip()
    if save_choice == 'y':
        save_tracked_points()
