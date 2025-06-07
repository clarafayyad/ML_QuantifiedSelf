# Video Frame Extractor

A comprehensive Python tool for extracting frames from video files using the powerful [video2images library](https://pypi.org/project/video2images/). This system provides both command-line and programmatic interfaces for converting videos to image sequences with advanced features like batch processing, time segment extraction, and custom configurations.

## 🎯 Features

- **Multiple Video Formats**: Support for `.mov`, `.avi`, `.mpg`, `.mpeg`, `.mp4`, `.mkv`, `.wmv`
- **Multiple Image Formats**: Output to `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.dicom`, `.dcm`
- **Time Segment Extraction**: Extract frames from specific time intervals
- **Custom Frame Rates**: Control capture rate (frames per second)
- **Batch Processing**: Process entire directories of videos
- **Video Information**: Get detailed video metadata
- **Configurable Settings**: JSON-based configuration system
- **Comprehensive Logging**: Detailed logging with multiple levels
- **Error Handling**: Robust error handling and validation

## 🚀 Quick Start

### Installation

1. **Create and activate a virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Basic usage**:
   ```bash
   # Extract all frames from a video
   python video_extractor.py input_video.mp4
   
   # Extract frames with custom settings
   python video_extractor.py video.mp4 --start-time 30 --end-time 120 --capture-rate 10
   ```

### Requirements

Based on the [video2images library](https://pypi.org/project/video2images/) requirements:
- `video2images>=1.3` - Core frame extraction library
- `opencv-python>=4.5.0` - Video processing and information extraction
- `tqdm>=4.60.0` - Progress bars (dependency of video2images)
- `imageio>=2.9.0` - Image I/O operations (dependency of video2images)
- `imageio-ffmpeg>=0.4.0` - FFmpeg support (dependency of video2images)
- `moviepy>=1.0.0` - Video processing (dependency of video2images)

## 📖 Usage Examples

### Command Line Interface

```bash
# Basic extraction - all frames
python video_extractor.py input_video.mp4

# Extract specific time segment (30s to 120s) at 5 fps
python video_extractor.py input_video.mp4 --start-time 30 --end-time 120 --capture-rate 5

# Custom output format and directory
python video_extractor.py input_video.mp4 --output-dir frames/ --save-format .png

# Batch process all videos in a directory
python video_extractor.py --batch-dir /path/to/videos/ --output-dir /path/to/output/

# Get video information only
python video_extractor.py --info input_video.mp4

# Create sample configuration file
python video_extractor.py --create-config
```

### Programmatic Usage

```python
from video_extractor import VideoFrameExtractor

# Initialize extractor
extractor = VideoFrameExtractor()

# Basic extraction
success, output_dir = extractor.extract_frames(
    video_path="input_video.mp4",
    output_dir="frames",
    save_format=".jpg"
)

# Time segment extraction
success, output_dir = extractor.extract_frames(
    video_path="input_video.mp4",
    start_time=30,      # Start at 30 seconds
    end_time=120,       # End at 120 seconds
    capture_rate=10,    # 10 frames per second
    save_format=".png"
)

# Batch processing
results = extractor.batch_extract(
    video_directory="videos/",
    output_base_dir="extracted_frames/",
    capture_rate=5
)

# Get video information
info = extractor.get_video_info("input_video.mp4")
if info:
    print(f"Duration: {info['duration_seconds']} seconds")
    print(f"FPS: {info['fps']}")
    print(f"Resolution: {info['width']}x{info['height']}")
```

## ⚙️ Configuration

### JSON Configuration File

Create a `video_extraction_config.json` file:

```json
{
    "default_capture_rate": 10,
    "default_save_format": ".jpg",
    "default_output_dir": "extracted_frames",
    "create_video_subfolders": true,
    "preserve_aspect_ratio": true,
    "log_level": "INFO"
}
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `default_capture_rate` | Default frames per second to capture | `null` (original FPS) |
| `default_save_format` | Default image format | `.jpg` |
| `default_output_dir` | Default output directory | `"extracted_frames"` |
| `create_video_subfolders` | Create separate folders for each video | `true` |
| `preserve_aspect_ratio` | Maintain original aspect ratio | `true` |
| `log_level` | Logging level (DEBUG, INFO, WARNING, ERROR) | `"INFO"` |

## 📁 File Structure

```
video_to_photo/
├── video_extractor.py          # Main extraction script
├── example_usage.py            # Example usage demonstrations
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── video_extraction.log        # Automatically created log file
└── extracted_frames/          # Default output directory
    ├── video1_20241203_143022/ # Video-specific subfolder
    │   ├── frame_001.jpg
    │   ├── frame_002.jpg
    │   └── ...
    └── video2_20241203_143055/
        ├── frame_001.jpg
        └── ...
```

## 🎬 Video Format Support

### Supported Input Formats
Based on [video2images library](https://pypi.org/project/video2images/) specifications:
- `.mov` - QuickTime Movie
- `.avi` - Audio Video Interleave
- `.mpg`, `.mpeg` - MPEG Video
- `.mp4` - MPEG-4 Video
- `.mkv` - Matroska Video
- `.wmv` - Windows Media Video

### Supported Output Formats
- `.jpg`, `.jpeg` - JPEG Image
- `.png` - Portable Network Graphics
- `.bmp` - Bitmap Image
- `.tiff`, `.tif` - Tagged Image File Format
- `.dicom`, `.dcm` - Digital Imaging and Communications in Medicine

## 🔧 Advanced Features

### Frame Capture Rate Control

The `capture_rate` parameter controls how many frames per second to extract:

```python
# Extract 5 frames per second (regardless of original video FPS)
extractor.extract_frames("video.mp4", capture_rate=5)

# Extract every frame (use original video FPS)
extractor.extract_frames("video.mp4", capture_rate=None)

# Extract 1 frame every 2 seconds
extractor.extract_frames("video.mp4", capture_rate=0.5)
```

### Time Segment Extraction

Extract frames from specific time intervals:

```python
# Extract frames from 1 minute to 3 minutes
extractor.extract_frames(
    "video.mp4",
    start_time=60,    # 1 minute
    end_time=180,     # 3 minutes
    capture_rate=10
)
```

### Batch Processing with Custom Parameters

```python
# Process all videos with consistent settings
results = extractor.batch_extract(
    video_directory="input_videos/",
    output_base_dir="batch_output/",
    start_time=10,
    end_time=60,
    capture_rate=5,
    save_format=".png"
)

# Check results
for video_path, (success, output_dir) in results.items():
    if success:
        print(f"✓ {os.path.basename(video_path)} -> {output_dir}")
    else:
        print(f"✗ Failed: {os.path.basename(video_path)}")
```

## 📊 Video Information Extraction

Get comprehensive video metadata:

```python
info = extractor.get_video_info("video.mp4")
print(f"""
Video Information:
- Filename: {info['filename']}
- Resolution: {info['width']}x{info['height']}
- Frame Rate: {info['fps']:.2f} fps
- Duration: {info['duration_seconds']:.1f} seconds
- Total Frames: {info['frame_count']}
- File Size: {info['size_mb']:.1f} MB
""")
```

## 🚨 Error Handling and Logging

The system includes comprehensive error handling and logging:

```python
# Check log file for detailed information
# Log file: video_extraction.log

# Example log output:
# 2024-12-03 14:30:22 - INFO - Starting frame extraction from: video.mp4
# 2024-12-03 14:30:22 - INFO - Output directory: extracted_frames/video_20241203_143022
# 2024-12-03 14:30:22 - INFO - Capture rate: 10 fps
# 2024-12-03 14:30:25 - INFO - ✓ Frame extraction completed successfully!
```

## 🔍 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'video2images'**
```bash
pip install video2images
```

**No video files found in directory**
- Check file extensions are supported
- Ensure files exist and are readable

**Low capture rate results in few frames**
- Increase `capture_rate` parameter
- Check video duration vs. time segment

**Large output directories**
- Reduce `capture_rate` for fewer frames
- Use shorter time segments
- Choose compressed formats like `.jpg`

### Performance Tips

- **Large Videos**: Use time segments instead of extracting all frames
- **Batch Processing**: Process videos in smaller batches to manage memory
- **Storage**: Use `.jpg` format for smaller file sizes, `.png` for quality
- **Speed**: Higher capture rates create more files but provide better temporal resolution

## 🔗 Related Projects

- [video2images](https://pypi.org/project/video2images/) - Core extraction library
- [OpenCV](https://opencv.org/) - Computer vision library for video processing
- [MoviePy](https://moviepy.readthedocs.io/) - Video editing and processing

## 📄 License

This project builds upon the [video2images library](https://pypi.org/project/video2images/) (MIT License) and is designed for research and educational purposes.

## 🤝 Contributing

Contributions welcome! Focus areas:
- Additional video format support
- Performance optimizations
- GUI interface development
- Integration with other video processing tools

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Based on**: [video2images v1.3](https://pypi.org/project/video2images/)  
**Author**: ML4QS Team 