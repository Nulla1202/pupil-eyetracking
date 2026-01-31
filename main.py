# -*- coding: utf-8 -*-
"""
GazeTrack - Real-time Eye Tracking Visualization
Main application entry point.

Usage:
    python main.py                      # Gaze tracking only
    python main.py --pupil              # Gaze + pupil measurement
    python main.py --pupil-only         # Pupil measurement only
    python main.py --offline video.mp4  # Offline video analysis

Controls:
    ESC or Q - Quit application
"""

import sys
import time
import argparse
import cv2

# Import configuration
import config

# Import components
from src.gaze_tracker import GazeTracker
from src.physics import PhysicsCalculator
from src.trail import TrailSystem
from src.renderer import Renderer
from src.pupil_detector import PupilDetector
from src.metrics import MetricsCalculator, GazeInfoMetricsCalculator
from src.offline_analyzer import run_offline_analysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GazeTrack - Real-time Eye Tracking Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Gaze tracking only
  python main.py --pupil              # Gaze + pupil measurement  
  python main.py --pupil-only         # Pupil measurement only (no gaze)
  python main.py --offline video.mp4  # Analyze video file offline
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--pupil", "-p",
        action="store_true",
        help="Enable pupil measurement alongside gaze tracking"
    )
    mode_group.add_argument(
        "--pupil-only",
        action="store_true",
        help="Pupil measurement only mode (no gaze tracking)"
    )
    mode_group.add_argument(
        "--offline",
        type=str,
        metavar="VIDEO",
        help="Analyze a video file offline for pupil metrics"
    )
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (simulated gaze for testing)"
    )
    
    # Display options
    parser.add_argument(
        "--windowed", "-w",
        action="store_true",
        help="Run in windowed mode instead of fullscreen"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.SCREEN_WIDTH,
        help="Window width (only used in windowed mode)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.SCREEN_HEIGHT,
        help="Window height (only used in windowed mode)"
    )
    
    # Tracking options
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip camera preview"
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration (not recommended for gaze tracking)"
    )
    
    # Output options
    parser.add_argument(
        "--save-data",
        type=str,
        default=None,
        help="Save gaze data to specified file on exit"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV file path for offline analysis"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file path for offline analysis"
    )
    
    # Pupil detection settings
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=config.EAR_THRESHOLD,
        help=f"EAR threshold for blink detection (default: {config.EAR_THRESHOLD})"
    )
    
    return parser.parse_args()


def run_demo_mode(args):
    """
    Run in demo mode with simulated gaze movement.
    Useful for testing visualization without a camera.
    """
    import math
    import random
    
    print("=" * 50)
    print("GazeTrack - Demo Mode")
    print("=" * 50)
    print("Running with simulated gaze data...")
    print("Press ESC or Q to quit")
    print("=" * 50)
    
    # Initialize components
    physics = PhysicsCalculator(
        smoothing_factor=config.SMOOTHING_FACTOR
    )
    
    trail = TrailSystem(
        max_length=config.TRAIL_MAX_LENGTH,
        min_alpha=config.TRAIL_MIN_ALPHA,
        max_alpha=config.TRAIL_MAX_ALPHA,
        min_radius=config.TRAIL_MIN_RADIUS,
        max_radius=config.TRAIL_MAX_RADIUS,
        base_color=config.GAZE_CIRCLE_COLOR
    )
    
    renderer = Renderer(
        width=args.width,
        height=args.height,
        fullscreen=not args.windowed,
        background_color=config.BACKGROUND_COLOR,
        gaze_color=config.GAZE_CIRCLE_COLOR,
        gaze_radius=config.GAZE_CIRCLE_RADIUS
    )
    
    if not renderer.initialize():
        print("Failed to initialize renderer!")
        return 1
    
    # Demo animation parameters
    start_time = time.time()
    center_x = renderer.width / 2
    center_y = renderer.height / 2
    
    running = True
    
    try:
        while running:
            # Handle events
            running = renderer.handle_events()
            
            # Generate simulated gaze position (figure-8 pattern)
            t = time.time() - start_time
            
            # Figure-8 / Lissajous curve
            x = center_x + math.sin(t * 0.8) * (renderer.width * 0.35)
            y = center_y + math.sin(t * 1.6) * (renderer.height * 0.25)
            
            # Add some noise for realism
            x += random.gauss(0, 5)
            y += random.gauss(0, 5)
            
            # Update physics
            gaze_state = physics.update(x, y)
            
            # Update trail
            trail.update(
                x=gaze_state.x,
                y=gaze_state.y,
                velocity_magnitude=gaze_state.velocity_magnitude,
                acceleration_magnitude=gaze_state.acceleration_magnitude,
                timestamp=gaze_state.timestamp
            )
            
            # Render
            renderer.render(gaze_state, trail, config.FPS_TARGET)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        renderer.cleanup()
    
    print("Demo mode ended.")
    return 0


def run_tracking_mode(args):
    """
    Run in full tracking mode with GazeFollower.
    """
    print("=" * 50)
    print("GazeTrack - Eye Tracking Visualization")
    print("=" * 50)
    print("Initializing GazeFollower...")
    
    # Initialize gaze tracker
    gaze_tracker = GazeTracker()
    
    if not gaze_tracker.initialize():
        print("Failed to initialize GazeFollower!")
        print("Make sure your webcam is connected and accessible.")
        return 1
    
    print("GazeFollower initialized successfully!")
    
    # Camera preview
    if not args.skip_preview:
        print("\nShowing camera preview...")
        print("Position yourself so your face is clearly visible.")
        print("Press any key to continue when ready.")
        gaze_tracker.preview()
    
    # Calibration
    if not args.skip_calibration:
        print("\nStarting calibration...")
        print("Follow the on-screen instructions.")
        if not gaze_tracker.calibrate():
            print("Calibration failed!")
            gaze_tracker.release()
            return 1
        print("Calibration complete!")
    else:
        print("\nWarning: Skipping calibration. Accuracy may be reduced.")
    
    # Initialize other components
    physics = PhysicsCalculator(
        smoothing_factor=config.SMOOTHING_FACTOR
    )
    
    trail = TrailSystem(
        max_length=config.TRAIL_MAX_LENGTH,
        min_alpha=config.TRAIL_MIN_ALPHA,
        max_alpha=config.TRAIL_MAX_ALPHA,
        min_radius=config.TRAIL_MIN_RADIUS,
        max_radius=config.TRAIL_MAX_RADIUS,
        base_color=config.GAZE_CIRCLE_COLOR
    )
    
    renderer = Renderer(
        width=args.width,
        height=args.height,
        fullscreen=not args.windowed,
        background_color=config.BACKGROUND_COLOR,
        gaze_color=config.GAZE_CIRCLE_COLOR,
        gaze_radius=config.GAZE_CIRCLE_RADIUS
    )
    
    if not renderer.initialize():
        print("Failed to initialize renderer!")
        gaze_tracker.release()
        return 1
    
    print("\nStarting gaze tracking...")
    print("Press ESC or Q to quit")
    print("=" * 50)
    
    # Start sampling
    gaze_tracker.start_sampling()
    
    running = True
    
    try:
        while running:
            # Handle events
            running = renderer.handle_events()
            
            # Get gaze position
            x, y, timestamp = gaze_tracker.get_gaze_position()
            
            # Update physics
            gaze_state = physics.update(x, y, timestamp)
            
            # Update trail
            trail.update(
                x=gaze_state.x,
                y=gaze_state.y,
                velocity_magnitude=gaze_state.velocity_magnitude,
                acceleration_magnitude=gaze_state.acceleration_magnitude,
                timestamp=gaze_state.timestamp
            )
            
            # Render
            renderer.render(gaze_state, trail, config.FPS_TARGET)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        gaze_tracker.stop_sampling()
        
        # Save data if requested
        if args.save_data:
            print(f"\nSaving gaze data to: {args.save_data}")
            gaze_tracker.save_data(args.save_data)
        
        gaze_tracker.release()
        renderer.cleanup()
    
    print("Application ended.")
    return 0


def run_gaze_with_pupil_mode(args):
    """
    Run gaze tracking with full pupil measurement (pupil diameter + blink detection).
    Uses GazeFollower's frame callback to receive camera frames for pupil detection.
    """
    print("=" * 50)
    print("GazeTrack - Gaze + Pupil Measurement")
    print("=" * 50)
    print("Initializing GazeFollower...")
    
    # Initialize gaze tracker
    gaze_tracker = GazeTracker()
    
    if not gaze_tracker.initialize():
        print("Failed to initialize GazeFollower!")
        print("Make sure your webcam is connected and accessible.")
        return 1
    
    print("GazeFollower initialized successfully!")
    
    # Camera preview (before enabling callback to not interfere with GazeFollower's internal display)
    if not args.skip_preview:
        print("\nShowing camera preview...")
        print("Position yourself so your face is clearly visible.")
        print("Press any key to continue when ready.")
        gaze_tracker.preview()
    
    # Calibration
    if not args.skip_calibration:
        print("\nStarting calibration...")
        print("Follow the on-screen instructions.")
        if not gaze_tracker.calibrate():
            print("Calibration failed!")
            gaze_tracker.release()
            return 1
        print("Calibration complete!")
    else:
        print("\nWarning: Skipping calibration. Gaze accuracy may be reduced.")
    
    # Enable frame callback AFTER calibration to not interfere with GazeFollower's preview/calibration
    print("Enabling frame callback for pupil detection...")
    # #region agent log
    import json; open(r'c:\Users\daist\gazetrack\.cursor\debug.log', 'a').write(json.dumps({"hypothesisId": "H6", "location": "main.py:after_calibration_enable_callback", "message": "enabling_callback_after_calibration", "timestamp": int(time.time()*1000)}) + '\n')
    # #endregion
    callback_result = gaze_tracker.enable_frame_callback()
    # #region agent log
    import json; open(r'c:\Users\daist\gazetrack\.cursor\debug.log', 'a').write(json.dumps({"hypothesisId": "H6", "location": "main.py:callback_result", "message": "callback_enable_result", "data": {"result": callback_result}, "timestamp": int(time.time()*1000)}) + '\n')
    # #endregion
    if not callback_result:
        print("Warning: Frame callback not available.")
    
    # Initialize pupil detector and metrics calculator
    print("Initializing pupil detector...")
    pupil_detector = PupilDetector()
    metrics_calc = MetricsCalculator(ear_threshold=args.ear_threshold)
    
    # Initialize other components
    physics = PhysicsCalculator(
        smoothing_factor=config.SMOOTHING_FACTOR
    )
    
    trail = TrailSystem(
        max_length=config.TRAIL_MAX_LENGTH,
        min_alpha=config.TRAIL_MIN_ALPHA,
        max_alpha=config.TRAIL_MAX_ALPHA,
        min_radius=config.TRAIL_MIN_RADIUS,
        max_radius=config.TRAIL_MAX_RADIUS,
        base_color=config.GAZE_CIRCLE_COLOR
    )
    
    renderer = Renderer(
        width=args.width,
        height=args.height,
        fullscreen=not args.windowed,
        background_color=config.BACKGROUND_COLOR,
        gaze_color=config.GAZE_CIRCLE_COLOR,
        gaze_radius=config.GAZE_CIRCLE_RADIUS
    )
    
    if not renderer.initialize():
        print("Failed to initialize renderer!")
        gaze_tracker.release()
        return 1
    
    print("\nStarting gaze + pupil tracking...")
    print("Press ESC or Q to quit")
    print("=" * 50)
    
    # Start sampling
    gaze_tracker.start_sampling()
    # #region agent log
    import json; open(r'c:\Users\daist\gazetrack\.cursor\debug.log', 'a').write(json.dumps({"hypothesisId": "H0", "location": "main.py:after_start_sampling", "message": "sampling_started", "timestamp": int(time.time()*1000)}) + '\n')
    # #endregion
    
    running = True
    loop_count = 0
    
    try:
        while running:
            loop_count += 1
            # Log first few iterations only
            if loop_count <= 3:
                # #region agent log
                import json; open(r'c:\Users\daist\gazetrack\.cursor\debug.log', 'a').write(json.dumps({"hypothesisId": "H0", "location": "main.py:main_loop", "message": "loop_iteration", "data": {"count": loop_count}, "timestamp": int(time.time()*1000)}) + '\n')
                # #endregion
            
            # Handle events
            running = renderer.handle_events()
            
            # Get gaze position
            x, y, timestamp = gaze_tracker.get_gaze_position()
            
            # Get frame via callback for pupil detection
            frame = gaze_tracker.get_frame()
            # #region agent log
            if loop_count <= 5:
                import json; open(r'c:\Users\daist\gazetrack\.cursor\debug.log', 'a').write(json.dumps({"hypothesisId": "H4_H5", "location": "main.py:run_gaze_with_pupil_mode", "message": "main_loop_frame", "data": {"frame_is_none": frame is None, "loop": loop_count}, "timestamp": int(time.time()*1000)}) + '\n')
            # #endregion
            if frame is not None:
                pupil_data = pupil_detector.process_frame(frame, timestamp)
                pupil_metrics = metrics_calc.update(pupil_data)
                ear_history = metrics_calc.get_ear_history()
                # #region agent log
                if loop_count <= 5:
                    import json; open(r'c:\Users\daist\gazetrack\.cursor\debug.log', 'a').write(json.dumps({"hypothesisId": "H4_H5", "location": "main.py:run_gaze_with_pupil_mode", "message": "pupil_processed", "data": {"has_metrics": pupil_metrics is not None, "left_diam": pupil_metrics.current_left_diameter_mm if pupil_metrics else None, "is_valid": pupil_metrics.is_valid if pupil_metrics else None, "loop": loop_count}, "timestamp": int(time.time()*1000)}) + '\n')
                # #endregion
            else:
                # Fallback: use GazeInfo for basic metrics
                pupil_metrics = None
                ear_history = None
            
            # Update physics
            gaze_state = physics.update(x, y, timestamp)
            
            # Update trail
            trail.update(
                x=gaze_state.x,
                y=gaze_state.y,
                velocity_magnitude=gaze_state.velocity_magnitude,
                acceleration_magnitude=gaze_state.acceleration_magnitude,
                timestamp=gaze_state.timestamp
            )
            
            # Render with pupil metrics
            renderer.render(
                gaze_state, trail, config.FPS_TARGET,
                pupil_metrics=pupil_metrics,
                ear_history=ear_history
            )
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        gaze_tracker.stop_sampling()
        
        # Save data if requested
        if args.save_data:
            print(f"\nSaving gaze data to: {args.save_data}")
            gaze_tracker.save_data(args.save_data)
        
        gaze_tracker.release()
        pupil_detector.release()
        renderer.cleanup()
    
    print("Application ended.")
    return 0


def run_pupil_only_mode(args):
    """
    Run pupil measurement only (no gaze tracking).
    """
    print("=" * 50)
    print("GazeTrack - Pupil Measurement Mode")
    print("=" * 50)
    print("Initializing pupil detector...")
    
    # Initialize pupil detection
    pupil_detector = PupilDetector()
    metrics_calc = MetricsCalculator(ear_threshold=args.ear_threshold)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam!")
        print("Make sure your webcam is connected and accessible.")
        return 1
    
    print("Webcam opened successfully!")
    
    # Initialize renderer
    renderer = Renderer(
        width=args.width,
        height=args.height,
        fullscreen=not args.windowed,
        background_color=config.BACKGROUND_COLOR,
        gaze_color=config.GAZE_CIRCLE_COLOR,
        gaze_radius=config.GAZE_CIRCLE_RADIUS
    )
    
    if not renderer.initialize():
        print("Failed to initialize renderer!")
        cap.release()
        return 1
    
    print("\nStarting pupil measurement...")
    print("Press ESC or Q to quit")
    print("=" * 50)
    
    running = True
    
    try:
        while running:
            # Handle events
            running = renderer.handle_events()
            
            # Get webcam frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            timestamp = time.time()
            
            # Process frame for pupil detection
            pupil_data = pupil_detector.process_frame(frame, timestamp)
            pupil_metrics = metrics_calc.update(pupil_data)
            ear_history = metrics_calc.get_ear_history()
            
            # Render pupil-only view
            renderer.render_pupil_only(
                config.FPS_TARGET,
                pupil_metrics=pupil_metrics,
                ear_history=ear_history
            )
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Save metrics data if requested
        if args.save_data:
            import json
            print(f"\nSaving pupil data to: {args.save_data}")
            data = metrics_calc.export_data()
            with open(args.save_data, 'w') as f:
                json.dump(data, f, indent=2)
        
        pupil_detector.release()
        cap.release()
        renderer.cleanup()
    
    print("Application ended.")
    return 0


def run_offline_mode(args):
    """
    Run offline video analysis.
    """
    print("=" * 50)
    print("GazeTrack - Offline Video Analysis")
    print("=" * 50)
    
    success = run_offline_analysis(
        video_path=args.offline,
        output_csv=args.output_csv,
        output_json=args.output_json,
        ear_threshold=args.ear_threshold
    )
    
    return 0 if success else 1


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.offline:
        return run_offline_mode(args)
    elif args.demo:
        return run_demo_mode(args)
    elif args.pupil_only:
        return run_pupil_only_mode(args)
    elif args.pupil:
        return run_gaze_with_pupil_mode(args)
    else:
        return run_tracking_mode(args)


if __name__ == "__main__":
    sys.exit(main())
