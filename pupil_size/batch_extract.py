import os
from pupil_measurement import get_pupil_measurements
import pandas as pd

image_folder = "../video_to_photo/frames/"
results = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        measurement = get_pupil_measurements(image_path)

        if measurement['success']:
            # Extract timestamp from filename like 'frame_12.5s.jpg'
            try:
                base = os.path.splitext(filename)[0]  # removes '.jpg'
                timestamp_str = base.replace("frame_", "").replace("s", "")
                timestamp = float(timestamp_str)
                measurement['timestamp'] = timestamp
            except ValueError:
                measurement['timestamp'] = None

            print("Processed {}".format(filename))
            results.append(measurement)

results_df = pd.DataFrame(results)
results_df.to_csv("pupil_size_extracted.csv", index=False)

print(f"Successfully processed {len(results)} images")
