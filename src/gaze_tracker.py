# -*- coding: utf-8 -*-
"""
GazeTracker - Wrapper class for GazeFollower library.
Handles initialization, calibration, and real-time gaze data retrieval.
"""

import time
import threading
from typing import Optional, Tuple, Any, Callable
import numpy as np
from gazefollower import GazeFollower


class GazeTracker:
    """
    Wrapper class for GazeFollower that manages eye tracking functionality.
    """
    
    def __init__(self):
        """Initialize the GazeTracker."""
        self._gaze_follower: Optional[GazeFollower] = None
        self._is_initialized = False
        self._is_sampling = False
        self._last_gaze: Tuple[float, float] = (0.0, 0.0)
        self._last_timestamp: float = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_callback_registered = False
        self._camera_original_callback: Optional[Callable] = None
        self._camera_callback_wrapper: Optional[Callable] = None
        self._camera_callback_warning_logged = False
    
    def initialize(self) -> bool:
        """
        Initialize the GazeFollower system.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            self._gaze_follower = GazeFollower()
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize GazeFollower: {e}")
            return False
    
    def preview(self) -> None:
        """
        Show camera preview to help user position themselves.
        """
        if self._gaze_follower is not None:
            self._gaze_follower.preview()
    
    def calibrate(self) -> bool:
        """
        Run the calibration procedure.
        
        Returns:
            bool: True if calibration was successful, False otherwise.
        """
        if self._gaze_follower is None:
            print("GazeFollower not initialized. Call initialize() first.")
            return False
        
        try:
            self._gaze_follower.calibrate()
            return True
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False
    
    def start_sampling(self) -> bool:
        """
        Start collecting gaze data.
        
        Returns:
            bool: True if sampling started successfully, False otherwise.
        """
        if self._gaze_follower is None:
            print("GazeFollower not initialized. Call initialize() first.")
            return False
        
        try:
            self._gaze_follower.start_sampling()
            self._is_sampling = True
            self._last_timestamp = time.time()
            return True
        except Exception as e:
            print(f"Failed to start sampling: {e}")
            return False
    
    def stop_sampling(self) -> None:
        """Stop collecting gaze data."""
        if self._gaze_follower is not None and self._is_sampling:
            self._gaze_follower.stop_sampling()
            self._is_sampling = False
    
    def get_gaze_position(self) -> Tuple[float, float, float]:
        """
        Get the current gaze position.

        Returns:
            Tuple[float, float, float]: (x, y, timestamp) of the current gaze position.
        """
        if self._gaze_follower is None or not self._is_sampling:
            return (self._last_gaze[0], self._last_gaze[1], time.time())

        try:
            # Get gaze info from GazeFollower
            gaze_info = self._gaze_follower.get_gaze_info()

            # Debug logging
            if gaze_info is None:
                print(f"DEBUG: gaze_info is None")
            else:
                print(f"DEBUG: gaze_info.status = {gaze_info.status}")
                if hasattr(gaze_info, 'filtered_gaze_coordinates'):
                    print(f"DEBUG: filtered_gaze_coordinates = {gaze_info.filtered_gaze_coordinates}")
                if hasattr(gaze_info, 'raw_gaze_coordinates'):
                    print(f"DEBUG: raw_gaze_coordinates = {gaze_info.raw_gaze_coordinates}")

            if gaze_info is not None and gaze_info.status:
                # Use filtered coordinates for smoothest tracking
                coords = gaze_info.filtered_gaze_coordinates
                if coords is not None and len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    print(f"DEBUG: Updated gaze position to ({x}, {y})")
                    self._last_gaze = (x, y)
                    self._last_timestamp = time.time()
                else:
                    print(f"DEBUG: coords is None or has insufficient length: {coords}")
            else:
                print(f"DEBUG: Using last known gaze position: {self._last_gaze}")

            return (self._last_gaze[0], self._last_gaze[1], self._last_timestamp)
        except Exception as e:
            # Log error for debugging but continue with last known position
            print(f"Warning: Failed to get gaze position: {e}")
            return (self._last_gaze[0], self._last_gaze[1], self._last_timestamp)

    def _on_frame_callback(self, camera_running_state, timestamp, frame: np.ndarray) -> None:
        """
        Callback function called by GazeFollower's camera when a new frame is captured.
        
        Args:
            camera_running_state: Current state of the camera
            timestamp: Timestamp of the frame
            frame: The captured frame (BGR format)
        """
        with self._frame_lock:
            self._last_frame = frame.copy() if frame is not None else None
    
    def enable_frame_callback(self) -> bool:
        """
        Enable frame callback to receive camera frames for pupil detection.
        Must be called before start_sampling().
        
        Returns:
            bool: True if callback was registered successfully
        """
        if self._gaze_follower is None:
            return False
        
        if self._frame_callback_registered:
            return True
        
        camera = getattr(self._gaze_follower, "camera", None)
        if camera is None:
            print("Warning: GazeFollower camera unavailable for callback registration.")
            return False

        # Prefer additive callback API if the SDK provides one so we don't interfere
        add_callback = getattr(camera, "add_on_image_callback", None)
        if callable(add_callback):
            try:
                add_callback(self._on_frame_callback)
                self._frame_callback_registered = True
                return True
            except Exception as e:
                print(f"Warning: Failed to add frame callback: {e}")
                return False

        set_callback = getattr(camera, "set_on_image_callback", None)
        if not callable(set_callback):
            print("Warning: Camera does not support setting image callbacks.")
            return False

        # Preserve original callback so gaze tracking continues to work
        callback_attr_candidates = [
            "callback_func",
            "_on_image_callback",
            "on_image_callback",
            "callback",
            "_callback"
        ]
        original_callback: Optional[Callable] = None
        original_callback_args = ()
        original_callback_kwargs = {}
        callback_attrs = [attr for attr in dir(camera) if "callback" in attr.lower()]
        for attr_name in callback_attr_candidates:
            candidate = getattr(camera, attr_name, None)
            if candidate is None or not callable(candidate):
                continue
            if candidate is set_callback or candidate is add_callback:
                continue
            original_callback = candidate
            original_callback_args = tuple(getattr(camera, "callback_args", ()) or ())
            original_callback_kwargs = dict(getattr(camera, "callback_kwargs", {}) or {})
            break

        def combined_callback(camera_state, timestamp, frame, *args, **kwargs):
            result = None
            if original_callback is not None:
                try:
                    result = original_callback(
                        camera_state,
                        timestamp,
                        frame,
                        *original_callback_args,
                        **original_callback_kwargs
                    )
                except Exception as e:
                    if not self._camera_callback_warning_logged:
                        print(f"Warning: Original camera callback failed: {e}")
                        self._camera_callback_warning_logged = True
            self._on_frame_callback(camera_state, timestamp, frame)
            return result

        try:
            set_callback(combined_callback)
            self._camera_original_callback = original_callback
            self._camera_callback_wrapper = combined_callback
            self._frame_callback_registered = True
            return True
        except Exception as e:
            print(f"Warning: Failed to register frame callback: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest camera frame received via callback.

        Returns:
            numpy array of the frame (BGR format) or None if unavailable.
        """
        with self._frame_lock:
            return self._last_frame.copy() if self._last_frame is not None else None
    
    def get_gaze_info(self) -> Any:
        """
        Get the full gaze info object from GazeFollower.
        
        Returns:
            GazeInfo object or None if unavailable.
        """
        if self._gaze_follower is None or not self._is_sampling:
            return None
        
        try:
            return self._gaze_follower.get_gaze_info()
        except Exception:
            return None
    
    def send_trigger(self, trigger_code: int) -> None:
        """
        Send a trigger/marker to the data stream.
        
        Args:
            trigger_code: Integer code for the trigger.
        """
        if self._gaze_follower is not None:
            self._gaze_follower.send_trigger(trigger_code)
    
    def save_data(self, filename: str) -> None:
        """
        Save collected gaze data to a file.
        
        Args:
            filename: Path to save the data file.
        """
        if self._gaze_follower is not None:
            self._gaze_follower.save_data(filename)
    
    def release(self) -> None:
        """Release all resources."""
        if self._gaze_follower is not None:
            self.stop_sampling()
            self._gaze_follower.release()
            self._gaze_follower = None
            self._is_initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if the tracker is initialized."""
        return self._is_initialized
    
    @property
    def is_sampling(self) -> bool:
        """Check if the tracker is currently sampling."""
        return self._is_sampling
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
