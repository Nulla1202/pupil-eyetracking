# -*- coding: utf-8 -*-
"""
Configuration settings for GazeTrack visualization system.
"""

# Display settings
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FULLSCREEN = True
FPS_TARGET = 60

# Gaze circle settings
GAZE_CIRCLE_RADIUS = 20
GAZE_CIRCLE_COLOR = (255, 50, 50)  # Red

# Trail settings
TRAIL_MAX_LENGTH = 30  # Number of trail points to keep
TRAIL_MIN_ALPHA = 20   # Minimum alpha for oldest trail point
TRAIL_MAX_ALPHA = 200  # Maximum alpha for newest trail point
TRAIL_MIN_RADIUS = 5   # Minimum radius for trail circles
TRAIL_MAX_RADIUS = 18  # Maximum radius for trail circles

# Physics settings
SMOOTHING_FACTOR = 0.3  # Low-pass filter factor (0-1, lower = more smoothing)
VELOCITY_SCALE = 1.0    # Scale factor for velocity display
ACCEL_SCALE = 1.0       # Scale factor for acceleration display

# Info panel settings
INFO_PANEL_PADDING = 20
INFO_PANEL_WIDTH = 280
INFO_PANEL_HEIGHT = 140
INFO_PANEL_BG_COLOR = (30, 30, 30)
INFO_PANEL_TEXT_COLOR = (220, 220, 220)
INFO_PANEL_ACCENT_COLOR = (255, 100, 100)
INFO_FONT_SIZE = 22

# Colors
BACKGROUND_COLOR = (15, 15, 25)

# =============================================================================
# Pupil Measurement Settings
# =============================================================================

# Pupil detection settings
PUPIL_MIN_DETECTION_CONFIDENCE = 0.5
PUPIL_MIN_TRACKING_CONFIDENCE = 0.5

# EAR (Eye Aspect Ratio) settings
EAR_THRESHOLD = 0.2  # Below this value, eyes are considered closed
EAR_CONSEC_FRAMES = 2  # Consecutive frames below threshold to confirm blink

# Pupil metrics display panel (left side)
PUPIL_PANEL_PADDING = 20
PUPIL_PANEL_WIDTH = 320
PUPIL_PANEL_HEIGHT = 280
PUPIL_PANEL_BG_COLOR = (30, 30, 40)
PUPIL_PANEL_TEXT_COLOR = (220, 220, 220)
PUPIL_PANEL_ACCENT_COLOR = (100, 200, 255)  # Cyan accent for pupil panel

# EAR graph settings
EAR_GRAPH_HEIGHT = 60
EAR_GRAPH_SAMPLES = 100
EAR_GRAPH_COLOR = (100, 255, 150)  # Green for EAR line
EAR_THRESHOLD_COLOR = (255, 100, 100)  # Red for threshold line

# Blink indicator
BLINK_INDICATOR_COLOR = (255, 255, 100)  # Yellow flash on blink

# Pupil diameter display
PUPIL_DIAMETER_COLOR_LEFT = (100, 150, 255)  # Blue for left eye
PUPIL_DIAMETER_COLOR_RIGHT = (255, 150, 100)  # Orange for right eye

