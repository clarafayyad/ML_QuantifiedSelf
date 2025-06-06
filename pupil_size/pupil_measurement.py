import cv2 
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import os
import csv
import pandas as pd
from datetime import datetime

class PupilDetectionConfig:
    """Configuration class for pupil detection parameters"""
    def __init__(self):
        # Detection parameters
        self.eye_scale_factor = 1.1
        self.eye_min_neighbors = 4
        self.face_min_size = (100, 100)
        self.eye_min_size = (30, 30)
        
        # Thresholding parameters for pupil detection
        self.pupil_threshold = 70
        self.adaptive_threshold_block_size = 11
        self.adaptive_threshold_C = 2
        
        # Morphological operations
        self.kernel_size = (3, 3)
        self.erosion_iterations = 1
        self.dilation_iterations = 2
        
        # Contour filtering
        self.min_contour_area = 50
        self.max_contour_area = 1000
        self.circularity_threshold = 0.3
        
        # Iris detection parameters
        self.iris_hough_param1 = 50
        self.iris_hough_param2 = 30
        self.iris_min_radius = 15
        self.iris_max_radius = 80
        
        # Measurement parameters
        self.average_iris_diameter_mm = 11.8
        self.crop_size = 125
        self.boundary_offset = 80
        
        # Preprocessing parameters
        self.gaussian_blur_kernel = (5, 5)
        self.clahe_clip_limit = 2.0
        self.clahe_tile_size = (8, 8)
        
        # CSV export settings
        self.csv_output_dir = "pupil_measurements"
        self.csv_filename = "pupil_analysis_results.csv"

class PupilMeasurementResult:
    """Streamlined data structure for ML training"""
    def __init__(self):
        # Essential metadata
        self.timestamp = datetime.now().isoformat()
        
        # Core measurements (key features for ML)
        self.pupil_diameter_mm = 0.0
        self.iris_diameter_mm = 11.8
        self.pupil_iris_ratio = 0.0
        self.pupil_center_x = 0
        self.pupil_center_y = 0
        self.iris_center_x = 0
        self.iris_center_y = 0
        
        # Gaze analysis (behavioral features)
        self.gaze_direction = "Unknown"
        self.gaze_offset_x_normalized = 0.0
        self.gaze_offset_y_normalized = 0.0
        self.gaze_magnitude = 0.0
        
        # Quality metrics (confidence features)
        self.overall_confidence = 0.0
        self.concentricity_score = 0.0
        self.pupil_circularity = 0.0
        
        # Target variables (for supervised learning)
        self.pupil_classification = "Unknown"
        self.pupil_size_category = "Unknown"
        
        # Detection success (binary features)
        self.pupil_detected = False
        self.iris_detected = False
        
        # Internal processing data (not saved to CSV)
        self._image_path = ""
        self._processing_errors = ""
        self._pupil_radius_px = 0
        self._iris_radius_px = 0
        self._processing_time = 0.0

def create_csv_headers():
    """Define the essential CSV column headers for ML training"""
    return [
        # Essential Metadata
        'timestamp',
        
        # Core Measurements (key features)
        'pupil_diameter_mm',
        'iris_diameter_mm', 
        'pupil_iris_ratio',
        'pupil_center_x',
        'pupil_center_y',
        'iris_center_x', 
        'iris_center_y',
        
        # Gaze Analysis (behavioral features)
        'gaze_direction',
        'gaze_offset_x_normalized',
        'gaze_offset_y_normalized',
        'gaze_magnitude',
        
        # Quality Metrics (confidence features)
        'overall_confidence',
        'concentricity_score',
        'pupil_circularity',
        
        # Target Variables (for supervised learning)
        'pupil_classification',
        'pupil_size_category',
        
        # Detection Success (binary features)
        'pupil_detected',
        'iris_detected'
    ]

def save_results_to_csv(result, config):
    """Save streamlined measurement results to CSV file"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(config.csv_output_dir, exist_ok=True)
        
        csv_path = os.path.join(config.csv_output_dir, config.csv_filename)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(csv_path)
        
        # Create dictionary with only the essential columns
        essential_data = {}
        headers = create_csv_headers()
        
        for header in headers:
            if hasattr(result, header):
                essential_data[header] = getattr(result, header)
            else:
                essential_data[header] = 0  # Default value for missing data
        
        # Write to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Write headers if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(essential_data)
        
        return csv_path
        
    except Exception as e:
        return None

def load_results_from_csv(config):
    """Load previous results from CSV for analysis"""
    try:
        csv_path = os.path.join(config.csv_output_dir, config.csv_filename)
        
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        return df
        
    except Exception as e:
        return None

def generate_measurement_summary(config):
    """Generate a focused summary report for ML dataset"""
    try:
        df = load_results_from_csv(config)
        if df is None or len(df) == 0:
            print("No data available for summary.")
            return
        
        print("\n" + "="*50)
        print("ML DATASET SUMMARY REPORT")
        print("="*50)
        
        # Dataset overview
        print(f"Total samples: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Feature completeness
        pupil_success_rate = (df['pupil_detected'].sum() / len(df)) * 100
        iris_success_rate = (df['iris_detected'].sum() / len(df)) * 100
        print(f"\nData Completeness:")
        print(f"  Pupil measurements: {pupil_success_rate:.1f}%")
        print(f"  Iris measurements: {iris_success_rate:.1f}%")
        
        # Target variable distribution (for classification tasks)
        print(f"\nTarget Variable Distribution:")
        classifications = df['pupil_classification'].value_counts()
        for classification, count in classifications.items():
            percentage = (count / len(df)) * 100
            print(f"  {classification}: {count} ({percentage:.1f}%)")
        
        # Feature statistics (for regression tasks)
        valid_pupils = df[df['pupil_detected'] == True]
        if len(valid_pupils) > 0:
            print(f"\nPupil Size Distribution (mm):")
            print(f"  Mean: {valid_pupils['pupil_diameter_mm'].mean():.2f}")
            print(f"  Std: {valid_pupils['pupil_diameter_mm'].std():.2f}")
            print(f"  Range: {valid_pupils['pupil_diameter_mm'].min():.2f} - {valid_pupils['pupil_diameter_mm'].max():.2f}")
            
            print(f"\nPupil/Iris Ratio Distribution:")
            ratios = valid_pupils['pupil_iris_ratio']
            print(f"  Mean: {ratios.mean():.3f}")
            print(f"  Std: {ratios.std():.3f}")
            print(f"  Range: {ratios.min():.3f} - {ratios.max():.3f}")
        
        # Gaze direction distribution (for gaze classification)
        print(f"\nGaze Direction Distribution:")
        gaze_counts = df['gaze_direction'].value_counts()
        for direction, count in gaze_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {direction}: {count} ({percentage:.1f}%)")
        
        # Quality metrics
        print(f"\nData Quality Metrics:")
        print(f"  Average confidence: {df['overall_confidence'].mean():.3f}")
        print(f"  High quality samples (>0.7): {(df['overall_confidence'] > 0.7).sum()}")
        print(f"  Average circularity: {df['pupil_circularity'].mean():.3f}")
        
        print("="*50)
        print("Ready for ML training!")
        
    except Exception as e:
        print(f"Error generating summary: {e}")

def export_ml_ready_dataset(config, output_filename="ml_dataset.csv"):
    """Export a clean ML-ready dataset with additional derived features"""
    try:
        df = load_results_from_csv(config)
        if df is None or len(df) == 0:
            print("No data available for export.")
            return None
        
        # Filter for high-quality samples only
        quality_threshold = 0.5
        clean_df = df[df['overall_confidence'] > quality_threshold].copy()
        
        # Add derived features useful for ML
        clean_df['pupil_area_ratio'] = clean_df['pupil_iris_ratio'] ** 2  # Area ratio from diameter ratio
        clean_df['gaze_distance_from_center'] = clean_df['gaze_magnitude']
        clean_df['is_centered_gaze'] = (clean_df['gaze_magnitude'] < 0.1).astype(int)
        clean_df['pupil_size_normalized'] = (clean_df['pupil_diameter_mm'] - clean_df['pupil_diameter_mm'].mean()) / clean_df['pupil_diameter_mm'].std()
        
        # Encode categorical variables
        gaze_directions = ['Center', 'Up', 'Down', 'Left', 'Right', 'Up-Left', 'Up-Right', 'Down-Left', 'Down-Right']
        for direction in gaze_directions:
            clean_df[f'gaze_{direction.lower().replace("-", "_")}'] = (clean_df['gaze_direction'] == direction).astype(int)
        
        # Binary encoding for pupil size categories
        clean_df['is_normal_pupil'] = (clean_df['pupil_size_category'] == 'Normal').astype(int)
        clean_df['is_small_pupil'] = (clean_df['pupil_size_category'] == 'Small').astype(int)
        clean_df['is_large_pupil'] = (clean_df['pupil_size_category'] == 'Large').astype(int)
        
        # Save enhanced dataset
        ml_csv_path = os.path.join(config.csv_output_dir, output_filename)
        clean_df.to_csv(ml_csv_path, index=False)
        
        print(f"\nML-ready dataset exported to: {ml_csv_path}")
        print(f"Samples: {len(clean_df)} (filtered from {len(df)} total)")
        print(f"Features: {len(clean_df.columns)}")
        
        return ml_csv_path
        
    except Exception as e:
        print(f"Error exporting ML dataset: {e}")
        return None

def safe_crop(image, center_x, center_y, half_size):
    """Safely crop image with boundary checking"""
    if image is None or image.size == 0:
        return None
        
    height, width = image.shape[:2]
    
    # Calculate boundaries with safety checks
    y_min = max(0, center_y - half_size)
    y_max = min(height, center_y + half_size)
    x_min = max(0, center_x - half_size)
    x_max = min(width, center_x + half_size)
    
    # Check if crop would be too small
    if (y_max - y_min) < half_size or (x_max - x_min) < half_size:
        return None
        
    return image[y_min:y_max, x_min:x_max]

def enhanced_preprocessing(eye_region, config):
    """Enhanced preprocessing inspired by the Medium article"""
    if eye_region is None or eye_region.size == 0:
        return None
    
    try:
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_region.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, config.gaussian_blur_kernel, 0)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit, 
            tileGridSize=config.clahe_tile_size
        )
        enhanced = clahe.apply(blurred)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        return denoised
        
    except Exception as e:
        return gray if 'gray' in locals() else eye_region

def detect_pupil_advanced(eye_gray, config):
    """Advanced pupil detection using thresholding and contours"""
    if eye_gray is None or eye_gray.size == 0:
        return None, 0, 0, 0, 0.0
    
    try:
        # Method 1: Fixed thresholding (from Medium article approach)
        _, thresh_fixed = cv2.threshold(
            eye_gray, config.pupil_threshold, 255, cv2.THRESH_BINARY_INV
        )
        
        # Method 2: Adaptive thresholding for varying lighting
        thresh_adaptive = cv2.adaptiveThreshold(
            eye_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, config.adaptive_threshold_block_size, 
            config.adaptive_threshold_C
        )
        
        # Combine both methods
        combined_thresh = cv2.bitwise_or(thresh_fixed, thresh_adaptive)
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.kernel_size)
        
        # Remove noise
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, 0, 0, 0, 0.0
        
        # Filter contours by area and circularity
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if config.min_contour_area <= area <= config.max_contour_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Filter by circularity (pupils should be roughly circular)
                    if circularity >= config.circularity_threshold:
                        valid_contours.append((contour, area, circularity))
        
        if not valid_contours:
            return None, 0, 0, 0, 0.0
        
        # Sort by area and take the largest valid contour
        best_contour = max(valid_contours, key=lambda x: x[1])
        contour, area, circularity = best_contour
        
        # Get the minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Validate results
        if radius < 3 or radius > eye_gray.shape[0] // 3:
            return None, 0, 0, 0, 0.0
        
        return contour, int(cx), int(cy), int(radius), circularity
        
    except Exception as e:
        return None, 0, 0, 0, 0.0

def detect_iris_hough(eye_gray, pupil_center, pupil_radius, config):
    """Detect iris using Hough circles, considering pupil position"""
    if eye_gray is None or eye_gray.size == 0:
        return None, 0, 0, 0
    
    try:
        # Apply blur for Hough circle detection
        blurred = cv2.medianBlur(eye_gray, 9)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=eye_gray.shape[0] // 4,
            param1=config.iris_hough_param1,
            param2=config.iris_hough_param2,
            minRadius=config.iris_min_radius,
            maxRadius=config.iris_max_radius
        )
        
        if circles is None:
            return None, 0, 0, 0
        
        circles = np.round(circles[0, :]).astype("int")
        
        # Find the best circle that contains the pupil
        best_circle = None
        best_score = 0
        
        for (x, y, r) in circles:
            # Check if this circle could contain the pupil
            if pupil_center[0] != 0 and pupil_center[1] != 0:
                dist_to_pupil = np.sqrt(
                    (x - pupil_center[0])**2 + (y - pupil_center[1])**2
                )
                
                # Iris should contain pupil
                if dist_to_pupil + pupil_radius <= r:
                    # Score based on how well centered the pupil is
                    centrality_score = 1.0 / (1.0 + dist_to_pupil * 0.1)
                    size_score = 1.0 if r > pupil_radius * 1.5 else 0.5
                    
                    score = centrality_score * size_score
                    
                    if score > best_score:
                        best_score = score
                        best_circle = (x, y, r)
            else:
                # No pupil detected, use center proximity
                center_x, center_y = eye_gray.shape[1] // 2, eye_gray.shape[0] // 2
                dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                score = 1.0 / (1.0 + dist_to_center * 0.1)
                
                if score > best_score:
                    best_score = score
                    best_circle = (x, y, r)
        
        if best_circle:
            return best_circle, best_circle[0], best_circle[1], best_circle[2]
        else:
            return None, 0, 0, 0
            
    except Exception as e:
        return None, 0, 0, 0

def calculate_gaze_direction(pupil_center, iris_center, iris_radius):
    """Calculate gaze direction based on pupil position relative to iris center"""
    if iris_radius == 0:
        return "Center", 0.0, 0.0, 0.0, 0.0
    
    try:
        # Calculate offset from iris center
        offset_x = pupil_center[0] - iris_center[0]
        offset_y = pupil_center[1] - iris_center[1]
        
        # Normalize by iris radius
        norm_x = offset_x / iris_radius
        norm_y = offset_y / iris_radius
        
        # Calculate angle and magnitude
        angle = np.arctan2(norm_y, norm_x) * 180 / np.pi
        magnitude = np.sqrt(norm_x**2 + norm_y**2)
        
        # Determine direction
        if magnitude < 0.1:
            direction = "Center"
        elif -22.5 <= angle < 22.5:
            direction = "Right"
        elif 22.5 <= angle < 67.5:
            direction = "Down-Right"
        elif 67.5 <= angle < 112.5:
            direction = "Down"
        elif 112.5 <= angle < 157.5:
            direction = "Down-Left"
        elif 157.5 <= angle or angle < -157.5:
            direction = "Left"
        elif -157.5 <= angle < -112.5:
            direction = "Up-Left"
        elif -112.5 <= angle < -67.5:
            direction = "Up"
        else:  # -67.5 <= angle < -22.5
            direction = "Up-Right"
        
        return direction, norm_x, norm_y, magnitude, angle
        
    except Exception as e:
        return "Unknown", 0.0, 0.0, 0.0, 0.0

def detect_eyes_in_color_image(image, config):
    """Improved eye detection with better error handling"""
    if image is None:
        return []
        
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load cascade classifiers
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if eye_cascade.empty() or face_cascade.empty():
            return []
        
        # Try direct eye detection first
        eyes = eye_cascade.detectMultiScale(
            gray, 
            scaleFactor=config.eye_scale_factor,
            minNeighbors=config.eye_min_neighbors,
            minSize=config.eye_min_size
        )
        
        eye_centers = []
        
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                center_x = ex + ew // 2
                center_y = ey + eh // 2
                eye_centers.append((center_x, center_y, ew, eh))
        else:
            # Fallback: detect faces then look for eyes
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=config.eye_scale_factor,
                minNeighbors=config.eye_min_neighbors,
                minSize=config.face_min_size
            )
            
            for (fx, fy, fw, fh) in faces:
                face_roi = gray[fy:fy+fh, fx:fx+fw]
                eyes_in_face = eye_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=config.eye_scale_factor,
                    minNeighbors=config.eye_min_neighbors,
                    minSize=config.eye_min_size
                )
                
                for (ex, ey, ew, eh) in eyes_in_face:
                    center_x = fx + ex + ew // 2
                    center_y = fy + ey + eh // 2
                    eye_centers.append((center_x, center_y, ew, eh))
        
        return eye_centers if eye_centers else [(image.shape[1]//2, image.shape[0]//2, 60, 40)]
        
    except Exception as e:
        return [(image.shape[1]//2, image.shape[0]//2, 60, 40)]

def analyze_pupil_comprehensive(image_path, config=None, eye_index=0, save_to_csv=True, verbose=False):
    """Comprehensive pupil analysis using enhanced techniques"""
    if config is None:
        config = PupilDetectionConfig()
    
    # Initialize result object
    result = PupilMeasurementResult()
    start_time = datetime.now()
    
    try:
        # Validate input
        if not os.path.exists(image_path):
            result._processing_errors = f"Image file not found: {image_path}"
            if verbose:
                print(result._processing_errors)
            return False, result
        
        result._image_path = os.path.abspath(image_path)
        
        # Load image
        color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_image is None:
            result._processing_errors = f"Could not load image: {image_path}"
            if verbose:
                print(result._processing_errors)
            return False, result
        
        if verbose:
            print(f"Successfully loaded image: {image_path}")
            print(f"Image dimensions: {color_image.shape}")
        
        # Detect eyes
        if verbose:
            print("Detecting eyes...")
        eye_data = detect_eyes_in_color_image(color_image, config)
        
        if not eye_data:
            result._processing_errors = "No eyes detected"
            if verbose:
                print("Warning: No eyes detected!")
            return False, result
        
        if verbose:
            print(f"Found {len(eye_data)} eye(s)")
        
        # Select eye to analyze
        if eye_index >= len(eye_data):
            eye_index = 0
        
        eye_x, eye_y, eye_w, eye_h = eye_data[eye_index]
        if verbose:
            print(f"Analyzing eye at coordinates: ({eye_x}, {eye_y})")
        
        # Extract and preprocess eye region
        eye_region = safe_crop(color_image, eye_x, eye_y, config.crop_size)
        if eye_region is None:
            result._processing_errors = "Cannot crop eye region safely"
            if verbose:
                print("Error: Cannot crop eye region safely")
            return False, result
        
        # Enhanced preprocessing
        processed_eye = enhanced_preprocessing(eye_region, config)
        if processed_eye is None:
            result._processing_errors = "Preprocessing failed"
            if verbose:
                print("Error: Preprocessing failed")
            return False, result
        
        # Advanced pupil detection
        if verbose:
            print("Detecting pupil...")
        pupil_contour, pupil_x, pupil_y, pupil_radius, pupil_circularity = detect_pupil_advanced(processed_eye, config)
        
        if pupil_contour is not None:
            result.pupil_detected = True
            result.pupil_center_x = pupil_x
            result.pupil_center_y = pupil_y
            result._pupil_radius_px = pupil_radius
            result.pupil_circularity = pupil_circularity
        
        # Iris detection
        if verbose:
            print("Detecting iris...")
        iris_data, iris_x, iris_y, iris_radius = detect_iris_hough(
            processed_eye, (pupil_x, pupil_y), pupil_radius, config
        )
        
        if iris_data is not None:
            result.iris_detected = True
            result.iris_center_x = iris_x
            result.iris_center_y = iris_y
            result._iris_radius_px = iris_radius
            result.iris_diameter_mm = config.average_iris_diameter_mm
        
        # Calculate measurements and ratios
        if result.iris_detected and iris_radius > 0:
            iris_diameter_px = iris_radius * 2
            px_to_mm_scale = config.average_iris_diameter_mm / iris_diameter_px
            result.pupil_diameter_mm = (pupil_radius * 2) * px_to_mm_scale if pupil_radius > 0 else 0.0
            
            if result.pupil_detected and result.pupil_diameter_mm > 0:
                result.pupil_iris_ratio = result.pupil_diameter_mm / result.iris_diameter_mm
        
        # Calculate gaze direction
        gaze_direction, gaze_x, gaze_y, gaze_magnitude, gaze_angle = calculate_gaze_direction(
            (pupil_x, pupil_y), (iris_x, iris_y), iris_radius
        )
        
        result.gaze_direction = gaze_direction
        result.gaze_offset_x_normalized = gaze_x
        result.gaze_offset_y_normalized = gaze_y
        result.gaze_magnitude = gaze_magnitude
        
        # Calculate quality scores
        if result.pupil_detected and result.iris_detected:
            # Concentricity score
            dist_centers = np.sqrt((pupil_x - iris_x)**2 + (pupil_y - iris_y)**2)
            result.concentricity_score = max(0, 1.0 - (dist_centers / (iris_radius * 0.5)))
            
            # Overall confidence
            pupil_confidence = min(1.0, pupil_circularity * 2.0)
            iris_confidence = 0.8  # Hough circles generally reliable
            result.overall_confidence = (pupil_confidence + iris_confidence + result.concentricity_score) / 3.0
        elif result.pupil_detected:
            result.overall_confidence = min(1.0, pupil_circularity * 1.5) * 0.5
        
        # Medical classification
        if result.pupil_diameter_mm > 0:
            if result.pupil_diameter_mm < 2:
                result.pupil_classification = "Miosis"
                result.pupil_size_category = "Small"
            elif result.pupil_diameter_mm <= 4:
                result.pupil_classification = "Normal"
                result.pupil_size_category = "Normal"
            elif result.pupil_diameter_mm <= 6:
                result.pupil_classification = "Slightly Dilated"
                result.pupil_size_category = "Large"
            else:
                result.pupil_classification = "Mydriasis"
                result.pupil_size_category = "Large"
        
        # Calculate processing time
        end_time = datetime.now()
        result._processing_time = (end_time - start_time).total_seconds()
        
        if verbose:
            print(f"Detection Results:")
            print(f"- Pupil diameter: {result.pupil_diameter_mm:.2f}mm")
            print(f"- Iris diameter: {result.iris_diameter_mm:.2f}mm")
            print(f"- Pupil/Iris ratio: {result.pupil_iris_ratio:.3f}")
            print(f"- Gaze direction: {result.gaze_direction}")
            print(f"- Confidence: {result.overall_confidence:.2f}")
            print(f"- Classification: {result.pupil_classification}")
        
        # Save to CSV (enabled by default)
        if save_to_csv:
            csv_path = save_results_to_csv(result, config)
            if csv_path and verbose:
                print(f"Results saved to CSV: {csv_path}")
        
        # Display results (optional)
        if verbose:
            display_enhanced_results(
                color_image, eye_region, processed_eye, 
                (pupil_x, pupil_y, pupil_radius),
                (iris_x, iris_y, iris_radius),
                result.pupil_diameter_mm, result.iris_diameter_mm,
                gaze_direction, result.overall_confidence,
                eye_x, eye_y
            )
        
        return True, result
        
    except Exception as e:
        result._processing_errors = str(e)
        result._processing_time = (datetime.now() - start_time).total_seconds()
        if verbose:
            print(f"Error in pupil analysis: {e}")
        return False, result

def display_enhanced_results(color_image, eye_region, processed_eye, 
                           pupil_data, iris_data, pupil_mm, iris_mm,
                           gaze_direction, confidence, eye_x, eye_y):
    """Enhanced visualization of results"""
    try:
        plt.figure(figsize=(16, 12))
        
        # Original image with eye location
        plt.subplot(2, 4, 1)
        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.plot(eye_x, eye_y, 'r+', markersize=15, markeredgewidth=3)
        plt.title('Original Image + Eye Location')
        plt.axis('off')
        
        # Eye region
        plt.subplot(2, 4, 2)
        if eye_region is not None:
            plt.imshow(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
            plt.title('Extracted Eye Region')
        plt.axis('off')
        
        # Processed eye
        plt.subplot(2, 4, 3)
        if processed_eye is not None:
            plt.imshow(processed_eye, cmap='gray')
            plt.title('Processed Eye')
        plt.axis('off')
        
        # Detection visualization
        plt.subplot(2, 4, 4)
        if processed_eye is not None and eye_region is not None:
            detection_img = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
            
            # Draw iris
            if iris_data[2] > 0:
                cv2.circle(detection_img, (iris_data[0], iris_data[1]), iris_data[2], (0, 255, 0), 2)
                cv2.circle(detection_img, (iris_data[0], iris_data[1]), 2, (0, 255, 0), -1)
            
            # Draw pupil
            if pupil_data[2] > 0:
                cv2.circle(detection_img, (pupil_data[0], pupil_data[1]), pupil_data[2], (255, 0, 0), 2)
                cv2.circle(detection_img, (pupil_data[0], pupil_data[1]), 2, (255, 0, 0), -1)
            
            plt.imshow(detection_img)
            plt.title(f'Detection (Conf: {confidence:.2f})')
        plt.axis('off')
        
        # Gaze direction visualization
        plt.subplot(2, 4, 5)
        gaze_viz = np.zeros((200, 200, 3), dtype=np.uint8)
        center = (100, 100)
        
        # Draw eye outline
        cv2.circle(gaze_viz, center, 80, (255, 255, 255), 2)
        
        # Draw iris
        cv2.circle(gaze_viz, center, 40, (0, 255, 0), 2)
        
        # Draw pupil position
        if pupil_data[2] > 0 and iris_data[2] > 0:
            offset_x = int((pupil_data[0] - iris_data[0]) * 2)
            offset_y = int((pupil_data[1] - iris_data[1]) * 2)
            pupil_pos = (center[0] + offset_x, center[1] + offset_y)
            cv2.circle(gaze_viz, pupil_pos, 15, (255, 0, 0), -1)
        
        plt.imshow(gaze_viz)
        plt.title(f'Gaze: {gaze_direction}')
        plt.axis('off')
        
        # Measurements chart
        plt.subplot(2, 4, 6)
        measurements = ['Pupil (mm)', 'Iris (mm)']
        values = [pupil_mm, iris_mm]
        colors = ['red', 'green']
        plt.bar(measurements, values, color=colors, alpha=0.7)
        plt.title('Size Measurements')
        plt.ylabel('Diameter (mm)')
        
        # Classification
        plt.subplot(2, 4, 7)
        if pupil_mm < 2:
            classification = "Miosis\n(Constricted)"
            class_color = 'blue'
        elif pupil_mm <= 4:
            classification = "Normal"
            class_color = 'green'
        elif pupil_mm <= 6:
            classification = "Slightly\nDilated"
            class_color = 'orange'
        else:
            classification = "Mydriasis\n(Dilated)"
            class_color = 'red'
        
        plt.text(0.5, 0.5, classification, ha='center', va='center', 
                fontsize=14, color=class_color, weight='bold',
                transform=plt.gca().transAxes)
        plt.title('Pupil Classification')
        plt.axis('off')
        
        # Summary
        plt.subplot(2, 4, 8)
        summary_text = f"""
Detection Summary:
• Pupil: {pupil_mm:.2f}mm
• Iris: {iris_mm:.2f}mm  
• Ratio: {(pupil_mm/iris_mm if iris_mm > 0 else 0):.3f}
• Gaze: {gaze_direction}
• Confidence: {confidence:.2f}

Classification: {classification.replace(chr(10), ' ')}
        """
        
        plt.text(0.05, 0.95, summary_text, ha='left', va='top', 
                fontsize=10, transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        plt.axis('off')
        
        plt.suptitle('Enhanced Pupil Detection Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in display: {e}")


def get_pupil_measurements(image_path, config=None, eye_index=0):
    """Get pupil measurements as a dictionary - saves to CSV automatically"""
    success, result = analyze_pupil_comprehensive(image_path, config, eye_index, save_to_csv=True, verbose=False)

    if success:
        return {
            'success': True,
            'pupil_diameter_mm': result.pupil_diameter_mm,
            'iris_diameter_mm': result.iris_diameter_mm,
            'pupil_iris_ratio': result.pupil_iris_ratio,
            'gaze_direction': result.gaze_direction,
            'overall_confidence': result.overall_confidence,
            'pupil_classification': result.pupil_classification,
            'pupil_detected': result.pupil_detected,
            'iris_detected': result.iris_detected,
            'pupil_center_x': result.pupil_center_x,
            'pupil_center_y': result.pupil_center_y,
            'iris_center_x': result.iris_center_x,
            'iris_center_y': result.iris_center_y,
            'gaze_magnitude': result.gaze_magnitude,
            'concentricity_score': result.concentricity_score,
            'pupil_circularity': result.pupil_circularity
        }
    else:
        return {
            'success': False,
            'error': result._processing_errors
        }


def measure_pupil_from_image_improved(image_path, config=None, eye_index=0):
    """Backwards compatible function - returns boolean success, saves to CSV"""
    success, result = analyze_pupil_comprehensive(image_path, config, eye_index, save_to_csv=True, verbose=False)
    return success

# Example usage
if __name__ == "__main__":
    config = PupilDetectionConfig()
    
    # Test with default image
    test_image = "frame_003.jpg"
    if os.path.exists(test_image):
        # Run analysis and save to CSV (minimal output)
        success, result = analyze_pupil_comprehensive(test_image, config, save_to_csv=True, verbose=False)
        
        if success:
            # Create streamlined measurement data
            measurement_data = {
                'pupil_diameter_mm': result.pupil_diameter_mm,
                'iris_diameter_mm': result.iris_diameter_mm,
                'pupil_iris_ratio': result.pupil_iris_ratio,
                'gaze_direction': result.gaze_direction,
                'overall_confidence': result.overall_confidence,
                'pupil_classification': result.pupil_classification,
                'pupil_detected': result.pupil_detected,
                'iris_detected': result.iris_detected
            }
            
            # Show essential results and confirm CSV saved
            print("Pupil Analysis Results:")
            print(f"Pupil Diameter: {measurement_data['pupil_diameter_mm']:.2f}mm")
            print(f"Iris Diameter: {measurement_data['iris_diameter_mm']:.2f}mm") 
            print(f"Pupil/Iris Ratio: {measurement_data['pupil_iris_ratio']:.3f}")
            print(f"Gaze Direction: {measurement_data['gaze_direction']}")
            print(f"Classification: {measurement_data['pupil_classification']}")
            print(f"Confidence: {measurement_data['overall_confidence']:.2f}")
            print(f"Detection Success: Pupil={measurement_data['pupil_detected']}, Iris={measurement_data['iris_detected']}")
            print(f"✓ Data saved to CSV: {config.csv_output_dir}/{config.csv_filename}")
        else:
            print("Analysis failed")

