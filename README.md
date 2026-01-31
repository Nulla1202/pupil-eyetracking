# Pupil Eyetracking

Real-time eye tracking visualization system with pupil measurement capabilities.

## Features

- **Real-time Gaze Tracking**: Track and visualize eye gaze position using webcam
- **Pupil Measurement**: Measure pupil diameter for both eyes in real-time
- **Blink Detection**: Detect blinks using Eye Aspect Ratio (EAR) algorithm
- **Trail Visualization**: Dynamic afterimage effects based on gaze velocity and acceleration
- **Physics-based Smoothing**: Low-pass filtering for smooth gaze visualization
- **Offline Analysis**: Process recorded video files for pupil metrics
- **Demo Mode**: Test visualization without a camera

## Requirements

- Python 3.8+
- Webcam (for real-time tracking)
- Windows/Linux/macOS

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pupil-eyetracking.git
cd pupil-eyetracking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Gaze Tracking
```bash
python main.py
```

### Gaze + Pupil Measurement
```bash
python main.py --pupil
```

### Pupil Measurement Only
```bash
python main.py --pupil-only
```

### Offline Video Analysis
```bash
python main.py --offline video.mp4 --output-csv results.csv
```

### Demo Mode (No Camera Required)
```bash
python main.py --demo
```

### Additional Options
```bash
python main.py --help

Options:
  --windowed, -w        Run in windowed mode instead of fullscreen
  --width WIDTH         Window width (default: 1920)
  --height HEIGHT       Window height (default: 1080)
  --skip-preview        Skip camera preview
  --skip-calibration    Skip calibration (not recommended)
  --save-data FILE      Save gaze data to file on exit
  --ear-threshold VAL   EAR threshold for blink detection (default: 0.2)
```

## Controls

- `ESC` or `Q` - Quit application

## Project Structure

```
pupil-eyetracking/
├── main.py              # Main application entry point
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── src/
    ├── gaze_tracker.py  # GazeFollower integration
    ├── physics.py       # Velocity/acceleration calculations
    ├── trail.py         # Trail visualization system
    ├── renderer.py      # Pygame-based rendering
    ├── pupil_detector.py    # Pupil detection with MediaPipe
    ├── metrics.py       # Pupil metrics calculation
    └── offline_analyzer.py  # Video file analysis
```

## Dependencies

- [gazefollower](https://pypi.org/project/gazefollower/) - Eye gaze tracking
- [pygame](https://www.pygame.org/) - Visualization rendering
- [numpy](https://numpy.org/) - Numerical computations
- [opencv-python](https://opencv.org/) - Image processing
- [mediapipe](https://mediapipe.dev/) - Face/eye landmark detection

## License

MIT License
