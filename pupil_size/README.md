# Pupil Detection & Analysis System

A comprehensive computer vision system for detecting and analyzing pupil measurements from eye images, optimized for machine learning applications and medical research.

## 🎯 Features

- **Advanced Pupil Detection**: Dual thresholding (fixed + adaptive) with contour-based analysis
- **Intelligent Iris Detection**: Hough circle detection with pupil-context awareness  
- **Gaze Direction Analysis**: 8-direction gaze classification based on pupil-iris positioning
- **Medical Classification**: Automatic pupil size categorization (Normal, Miosis, Mydriasis)
- **ML-Ready Data Export**: Streamlined 19-column CSV format optimized for machine learning
- **Quality Assessment**: Multi-factor confidence scoring and validation
- **Silent Operation**: Clean processing with minimal terminal output

## 🚀 Quick Start

### Installation

1. **Clone/Download** the repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run analysis**:
   ```bash
   python pupil_measurement.py
   ```

### Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy 
- Matplotlib
- Pandas (for CSV analysis)

## 📊 Usage Examples

### Basic Usage

```python
from pupil_measurement import get_pupil_measurements

# Analyze an image and get measurements
results = get_pupil_measurements("path/to/eye_image.jpg")

if results['success']:
    print(f"Pupil diameter: {results['pupil_diameter_mm']:.2f}mm")
    print(f"Classification: {results['pupil_classification']}")
    print(f"Gaze direction: {results['gaze_direction']}")
```

### Advanced Analysis

```python
from pupil_measurement import analyze_pupil_comprehensive, PupilDetectionConfig

# Custom configuration
config = PupilDetectionConfig()
config.pupil_threshold = 80  # Adjust detection sensitivity

# Full analysis with options
success, result = analyze_pupil_comprehensive(
    image_path="eye_image.jpg",
    config=config,
    save_to_csv=True,    # Save to CSV
    verbose=True         # Show detailed output
)
```

### Batch Processing

```python
import os
from pupil_measurement import get_pupil_measurements

# Process multiple images
image_folder = "path/to/images/"
results = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        measurement = get_pupil_measurements(image_path)
        if measurement['success']:
            results.append(measurement)

print(f"Successfully processed {len(results)} images")
```

## 📈 Output Data Structure

### Console Output
```
Pupil Analysis Results:
Pupil Diameter: 3.25mm
Iris Diameter: 11.80mm
Pupil/Iris Ratio: 0.276
Gaze Direction: Center
Classification: Normal
Confidence: 0.84
Detection Success: Pupil=True, Iris=True
✓ Data saved to CSV: pupil_measurements/pupil_analysis_results.csv
```

### CSV Data Structure (19 Essential Columns)

| Category | Columns | Description |
|----------|---------|-------------|
| **Metadata** | `timestamp` | Analysis timestamp |
| **Core Measurements** | `pupil_diameter_mm`, `iris_diameter_mm`, `pupil_iris_ratio` | Primary size measurements |
| **Spatial Data** | `pupil_center_x`, `pupil_center_y`, `iris_center_x`, `iris_center_y` | Anatomical positions |
| **Gaze Analysis** | `gaze_direction`, `gaze_offset_x_normalized`, `gaze_offset_y_normalized`, `gaze_magnitude` | Behavioral features |
| **Quality Metrics** | `overall_confidence`, `concentricity_score`, `pupil_circularity` | Confidence features |
| **Classification** | `pupil_classification`, `pupil_size_category` | Target variables |
| **Detection Status** | `pupil_detected`, `iris_detected` | Binary success flags |

## 🧠 Machine Learning Applications

### Classification Tasks
- **Pupil Size Classification**: Normal, Miosis, Mydriasis, Slightly Dilated
- **Gaze Direction**: 9 directions (Center, Up, Down, Left, Right, Up-Left, Up-Right, Down-Left, Down-Right)
- **Medical Screening**: Automated pupil abnormality detection

### Regression Tasks
- **Pupil Size Prediction**: Continuous diameter measurements
- **Gaze Magnitude**: Quantified gaze displacement
- **Quality Assessment**: Confidence scoring for measurement reliability

### Feature Engineering
```python
# Example feature extraction for ML
import pandas as pd

# Load data
df = pd.read_csv('pupil_measurements/pupil_analysis_results.csv')

# Derived features
df['pupil_area_ratio'] = df['pupil_iris_ratio'] ** 2
df['is_centered_gaze'] = (df['gaze_magnitude'] < 0.1).astype(int)
df['size_category_encoded'] = df['pupil_size_category'].map({
    'Small': 0, 'Normal': 1, 'Large': 2
})
```

## 🔧 Configuration Options

### Detection Parameters
```python
config = PupilDetectionConfig()

# Thresholding
config.pupil_threshold = 70                    # Fixed threshold value
config.adaptive_threshold_block_size = 11      # Adaptive block size
config.adaptive_threshold_C = 2                # Adaptive threshold constant

# Contour filtering  
config.min_contour_area = 50                   # Minimum pupil area (pixels)
config.max_contour_area = 1000                 # Maximum pupil area (pixels)
config.circularity_threshold = 0.3             # Minimum circularity (0-1)

# Iris detection
config.iris_hough_param1 = 50                  # Hough gradient threshold
config.iris_hough_param2 = 30                  # Hough accumulator threshold
config.iris_min_radius = 15                    # Minimum iris radius
config.iris_max_radius = 80                    # Maximum iris radius
```

## 📁 File Structure

```
pupil_size/
├── pupil_measurement.py          # Main analysis script
├── requirements.txt               # Python dependencies  
├── README.md                     # This documentation
├── 2019-11-22-17-33-27_Color.jpeg # Sample test image
├── pupil_measurements/           # Output directory
│   ├── pupil_analysis_results.csv  # Raw measurement data
│   └── ml_dataset.csv              # Enhanced ML-ready dataset
└── venv/                         # Python virtual environment
```

## 🔬 Technical Details

### Detection Algorithm
1. **Preprocessing**: CLAHE enhancement, Gaussian blur, denoising
2. **Pupil Detection**: Dual thresholding + contour analysis with circularity filtering
3. **Iris Detection**: Hough circle detection with pupil-context validation
4. **Gaze Analysis**: Geometric calculation of pupil position relative to iris center
5. **Quality Assessment**: Multi-factor confidence scoring based on concentricity and shape

### Medical Classifications
- **Normal**: 2.0-4.0mm diameter
- **Miosis** (Constricted): <2.0mm diameter  
- **Slightly Dilated**: 4.0-6.0mm diameter
- **Mydriasis** (Dilated): >6.0mm diameter

### Gaze Directions
- **9 Zones**: Center, Up, Down, Left, Right, Up-Left, Up-Right, Down-Left, Down-Right
- **Threshold**: <0.1 normalized magnitude for "Center" classification
- **Calculation**: Based on pupil offset from iris center, normalized by iris radius

## 📊 Data Analysis Functions

### Summary Statistics
```python
from pupil_measurement import generate_measurement_summary, PupilDetectionConfig

config = PupilDetectionConfig()
generate_measurement_summary(config)
```

### Enhanced Dataset Export
```python
from pupil_measurement import export_ml_ready_dataset

# Export enhanced dataset with derived features
ml_path = export_ml_ready_dataset(config, "enhanced_dataset.csv")
```

## 🎯 Research Applications

- **Medical Diagnostics**: Automated pupil response assessment
- **Neurological Research**: Pupil dynamics and brain function correlation
- **Human-Computer Interaction**: Gaze-based interface development
- **Computer Vision**: Eye tracking and attention analysis
- **Machine Learning**: Biometric feature extraction and classification

## 🔧 Troubleshooting

### Common Issues

**No eyes detected**
- Ensure good image quality and lighting
- Try adjusting `config.eye_scale_factor` (default: 1.1)

**Poor pupil detection**
- Adjust `config.pupil_threshold` (try 60-90 range)
- Modify `config.circularity_threshold` for stricter/looser filtering

**Low confidence scores**
- Check image resolution and focus
- Ensure pupil is clearly visible and unobstructed

### Performance Tips

- **Image Quality**: Use high-resolution, well-lit images
- **Preprocessing**: Images with good contrast work best
- **Batch Processing**: Process multiple images for statistical reliability

## 📄 License

This project is designed for research and educational purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

Contributions welcome! Please focus on:
- Algorithm improvements
- Additional ML features  
- Documentation enhancements
- Performance optimizations

---

**Version**: 2.0  
**Last Updated**: December 2024  
**Author**: ML4QS Team 