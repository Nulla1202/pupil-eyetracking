# -*- coding: utf-8 -*-
"""
Metrics Calculator - Processes pupil data to compute blink detection,
statistics, and time-series analysis.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import numpy as np

from .pupil_detector import PupilData


@dataclass
class BlinkEvent:
    """Represents a detected blink event."""
    start_time: float
    end_time: float
    duration: float
    min_ear: float


@dataclass
class PupilMetrics:
    """Aggregated metrics from pupil measurements."""
    # Current values
    current_left_diameter_mm: float
    current_right_diameter_mm: float
    current_avg_diameter_mm: float
    current_ear: float
    
    # Statistics (over buffer window)
    avg_left_diameter_mm: float
    avg_right_diameter_mm: float
    std_left_diameter_mm: float
    std_right_diameter_mm: float
    
    # Blink state
    is_blinking: bool
    blink_count: int
    blinks_per_minute: float
    last_blink_duration: float
    
    # Status
    is_valid: bool
    timestamp: float


class MetricsCalculator:
    """
    Calculates pupil metrics including blink detection from PupilData stream.
    """
    
    def __init__(
        self,
        buffer_size: int = 300,  # ~5 seconds at 60fps
        ear_threshold: float = 0.2,
        ear_consec_frames: int = 2,
        blink_rate_window: float = 60.0  # seconds for blink rate calculation
    ):
        """
        Initialize the metrics calculator.
        
        Args:
            buffer_size: Number of samples to keep for statistics
            ear_threshold: EAR threshold below which eyes are considered closed
            ear_consec_frames: Consecutive frames below threshold to confirm blink
            blink_rate_window: Time window in seconds for blink rate calculation
        """
        self._buffer_size = buffer_size
        self._ear_threshold = ear_threshold
        self._ear_consec_frames = ear_consec_frames
        self._blink_rate_window = blink_rate_window
        
        # Data buffers
        self._left_diameter_buffer: deque = deque(maxlen=buffer_size)
        self._right_diameter_buffer: deque = deque(maxlen=buffer_size)
        self._ear_buffer: deque = deque(maxlen=buffer_size)
        self._timestamp_buffer: deque = deque(maxlen=buffer_size)
        
        # Blink detection state
        self._frames_below_threshold = 0
        self._is_blinking = False
        self._blink_start_time: Optional[float] = None
        self._blink_min_ear: float = 1.0
        
        # Blink history
        self._blinks: List[BlinkEvent] = []
        self._last_blink_duration: float = 0.0
        
        # Start time for blink rate
        self._start_time: Optional[float] = None
    
    def update(self, pupil_data: PupilData) -> PupilMetrics:
        """
        Update metrics with new pupil data.
        
        Args:
            pupil_data: Latest pupil detection data
            
        Returns:
            PupilMetrics with current and aggregated values
        """
        timestamp = pupil_data.timestamp
        
        # Initialize start time
        if self._start_time is None:
            self._start_time = timestamp
        
        # Add to buffers
        if pupil_data.is_valid:
            self._left_diameter_buffer.append(pupil_data.left_diameter_mm)
            self._right_diameter_buffer.append(pupil_data.right_diameter_mm)
            self._ear_buffer.append(pupil_data.avg_ear)
            self._timestamp_buffer.append(timestamp)
        
        # Process blink detection
        self._detect_blink(pupil_data.avg_ear, timestamp)
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        # Calculate blink rate
        blink_rate = self._calculate_blink_rate(timestamp)
        
        # Current values
        current_avg = (pupil_data.left_diameter_mm + pupil_data.right_diameter_mm) / 2.0
        
        return PupilMetrics(
            current_left_diameter_mm=pupil_data.left_diameter_mm,
            current_right_diameter_mm=pupil_data.right_diameter_mm,
            current_avg_diameter_mm=current_avg,
            current_ear=pupil_data.avg_ear,
            avg_left_diameter_mm=stats[0],
            avg_right_diameter_mm=stats[1],
            std_left_diameter_mm=stats[2],
            std_right_diameter_mm=stats[3],
            is_blinking=self._is_blinking,
            blink_count=len(self._blinks),
            blinks_per_minute=blink_rate,
            last_blink_duration=self._last_blink_duration,
            is_valid=pupil_data.is_valid,
            timestamp=timestamp
        )
    
    def _detect_blink(self, ear: float, timestamp: float) -> None:
        """
        Detect blinks based on EAR threshold.
        
        A blink is detected when EAR drops below threshold for
        consecutive frames and then rises back up.
        """
        if ear < self._ear_threshold:
            self._frames_below_threshold += 1
            self._blink_min_ear = min(self._blink_min_ear, ear)
            
            if self._frames_below_threshold >= self._ear_consec_frames:
                if not self._is_blinking:
                    # Blink started
                    self._is_blinking = True
                    self._blink_start_time = timestamp
        else:
            if self._is_blinking and self._blink_start_time is not None:
                # Blink ended
                duration = timestamp - self._blink_start_time
                
                # Only count as blink if duration is reasonable (50ms to 500ms)
                if 0.05 <= duration <= 0.5:
                    blink = BlinkEvent(
                        start_time=self._blink_start_time,
                        end_time=timestamp,
                        duration=duration,
                        min_ear=self._blink_min_ear
                    )
                    self._blinks.append(blink)
                    self._last_blink_duration = duration * 1000  # Convert to ms
            
            # Reset state
            self._is_blinking = False
            self._frames_below_threshold = 0
            self._blink_min_ear = 1.0
            self._blink_start_time = None
    
    def _calculate_statistics(self) -> Tuple[float, float, float, float]:
        """
        Calculate statistics from buffers.
        
        Returns:
            Tuple of (avg_left, avg_right, std_left, std_right)
        """
        if len(self._left_diameter_buffer) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        
        left_arr = np.array(self._left_diameter_buffer)
        right_arr = np.array(self._right_diameter_buffer)
        
        return (
            float(np.mean(left_arr)),
            float(np.mean(right_arr)),
            float(np.std(left_arr)),
            float(np.std(right_arr))
        )
    
    def _calculate_blink_rate(self, current_time: float) -> float:
        """
        Calculate blinks per minute over the rate window.
        """
        if not self._blinks:
            return 0.0
        
        # Count blinks within the window
        window_start = current_time - self._blink_rate_window
        recent_blinks = [b for b in self._blinks if b.end_time >= window_start]
        
        # Calculate elapsed time (capped at window size)
        if self._start_time is None:
            return 0.0
        
        elapsed = min(current_time - self._start_time, self._blink_rate_window)
        if elapsed <= 0:
            return 0.0
        
        # Blinks per minute
        blink_rate = (len(recent_blinks) / elapsed) * 60.0
        return blink_rate
    
    def get_ear_history(self, num_samples: int = 100) -> List[float]:
        """
        Get recent EAR values for plotting.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of recent EAR values
        """
        ear_list = list(self._ear_buffer)
        return ear_list[-num_samples:] if len(ear_list) > num_samples else ear_list
    
    def get_diameter_history(self, num_samples: int = 100) -> Tuple[List[float], List[float]]:
        """
        Get recent diameter values for plotting.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            Tuple of (left_diameters, right_diameters)
        """
        left = list(self._left_diameter_buffer)
        right = list(self._right_diameter_buffer)
        
        left = left[-num_samples:] if len(left) > num_samples else left
        right = right[-num_samples:] if len(right) > num_samples else right
        
        return (left, right)
    
    def get_blink_events(self) -> List[BlinkEvent]:
        """Get all recorded blink events."""
        return self._blinks.copy()
    
    def reset(self) -> None:
        """Reset all buffers and state."""
        self._left_diameter_buffer.clear()
        self._right_diameter_buffer.clear()
        self._ear_buffer.clear()
        self._timestamp_buffer.clear()
        self._blinks.clear()
        self._frames_below_threshold = 0
        self._is_blinking = False
        self._blink_start_time = None
        self._blink_min_ear = 1.0
        self._last_blink_duration = 0.0
        self._start_time = None
    
    def export_data(self) -> dict:
        """
        Export all collected data as a dictionary.
        
        Returns:
            Dictionary with timestamps, diameters, EAR values, and blink events
        """
        return {
            'timestamps': list(self._timestamp_buffer),
            'left_diameter_mm': list(self._left_diameter_buffer),
            'right_diameter_mm': list(self._right_diameter_buffer),
            'ear': list(self._ear_buffer),
            'blinks': [
                {
                    'start_time': b.start_time,
                    'end_time': b.end_time,
                    'duration_ms': b.duration * 1000,
                    'min_ear': b.min_ear
                }
                for b in self._blinks
            ]
        }


class GazeInfoMetricsCalculator:
    """
    Simplified metrics calculator that uses GazeInfo's eye openness data directly.
    This avoids the need to access camera frames separately when GazeFollower is running.
    """
    
    def __init__(
        self,
        buffer_size: int = 300,
        openness_threshold: float = 0.3,  # Below this, eyes considered closed
        consec_frames: int = 2,
        blink_rate_window: float = 60.0
    ):
        """
        Initialize the GazeInfo metrics calculator.
        
        Args:
            buffer_size: Number of samples to keep for statistics
            openness_threshold: Eye openness threshold for blink detection
            consec_frames: Consecutive frames below threshold to confirm blink
            blink_rate_window: Time window in seconds for blink rate calculation
        """
        self._buffer_size = buffer_size
        self._openness_threshold = openness_threshold
        self._consec_frames = consec_frames
        self._blink_rate_window = blink_rate_window
        
        # Data buffers
        self._left_openness_buffer: deque = deque(maxlen=buffer_size)
        self._right_openness_buffer: deque = deque(maxlen=buffer_size)
        self._timestamp_buffer: deque = deque(maxlen=buffer_size)
        
        # Blink detection state
        self._frames_below_threshold = 0
        self._is_blinking = False
        self._blink_start_time: Optional[float] = None
        self._blink_min_openness: float = 1.0
        
        # Blink history
        self._blinks: List[BlinkEvent] = []
        self._last_blink_duration: float = 0.0
        
        # Start time for blink rate
        self._start_time: Optional[float] = None
    
    def update_from_gaze_info(self, gaze_info: Any, timestamp: float) -> PupilMetrics:
        """
        Update metrics using GazeInfo from GazeFollower.
        
        Args:
            gaze_info: GazeInfo object from GazeFollower
            timestamp: Current timestamp
            
        Returns:
            PupilMetrics with available data (diameters will be 0 as not available)
        """
        # Initialize start time
        if self._start_time is None:
            self._start_time = timestamp
        
        # Get eye openness values (these are already computed by GazeFollower)
        left_openness = 0.0
        right_openness = 0.0
        is_valid = False
        
        if gaze_info is not None and gaze_info.status:
            left_openness = float(gaze_info.left_openness) if gaze_info.left_openness else 0.0
            right_openness = float(gaze_info.right_openness) if gaze_info.right_openness else 0.0
            is_valid = True
            
            # Add to buffers
            self._left_openness_buffer.append(left_openness)
            self._right_openness_buffer.append(right_openness)
            self._timestamp_buffer.append(timestamp)
        
        # Average openness (use as EAR proxy)
        avg_openness = (left_openness + right_openness) / 2.0
        
        # Process blink detection using openness
        self._detect_blink(avg_openness, timestamp)
        
        # Calculate blink rate
        blink_rate = self._calculate_blink_rate(timestamp)
        
        # Create metrics (pupil diameter not available in this mode)
        return PupilMetrics(
            current_left_diameter_mm=0.0,  # Not available without frame processing
            current_right_diameter_mm=0.0,
            current_avg_diameter_mm=0.0,
            current_ear=avg_openness,  # Use openness as EAR proxy
            avg_left_diameter_mm=0.0,
            avg_right_diameter_mm=0.0,
            std_left_diameter_mm=0.0,
            std_right_diameter_mm=0.0,
            is_blinking=self._is_blinking,
            blink_count=len(self._blinks),
            blinks_per_minute=blink_rate,
            last_blink_duration=self._last_blink_duration,
            is_valid=is_valid,
            timestamp=timestamp
        )
    
    def _detect_blink(self, openness: float, timestamp: float) -> None:
        """Detect blinks based on eye openness threshold."""
        if openness < self._openness_threshold:
            self._frames_below_threshold += 1
            self._blink_min_openness = min(self._blink_min_openness, openness)
            
            if self._frames_below_threshold >= self._consec_frames:
                if not self._is_blinking:
                    self._is_blinking = True
                    self._blink_start_time = timestamp
        else:
            if self._is_blinking and self._blink_start_time is not None:
                duration = timestamp - self._blink_start_time
                
                if 0.05 <= duration <= 0.5:
                    blink = BlinkEvent(
                        start_time=self._blink_start_time,
                        end_time=timestamp,
                        duration=duration,
                        min_ear=self._blink_min_openness
                    )
                    self._blinks.append(blink)
                    self._last_blink_duration = duration * 1000
            
            self._is_blinking = False
            self._frames_below_threshold = 0
            self._blink_min_openness = 1.0
            self._blink_start_time = None
    
    def _calculate_blink_rate(self, current_time: float) -> float:
        """Calculate blinks per minute over the rate window."""
        if not self._blinks:
            return 0.0
        
        window_start = current_time - self._blink_rate_window
        recent_blinks = [b for b in self._blinks if b.end_time >= window_start]
        
        if self._start_time is None:
            return 0.0
        
        elapsed = min(current_time - self._start_time, self._blink_rate_window)
        if elapsed <= 0:
            return 0.0
        
        return (len(recent_blinks) / elapsed) * 60.0
    
    def get_openness_history(self, num_samples: int = 100) -> List[float]:
        """Get recent eye openness values for plotting."""
        # Return average of left and right
        if not self._left_openness_buffer:
            return []
        
        left = list(self._left_openness_buffer)[-num_samples:]
        right = list(self._right_openness_buffer)[-num_samples:]
        
        return [(l + r) / 2.0 for l, r in zip(left, right)]
    
    def reset(self) -> None:
        """Reset all buffers and state."""
        self._left_openness_buffer.clear()
        self._right_openness_buffer.clear()
        self._timestamp_buffer.clear()
        self._blinks.clear()
        self._frames_below_threshold = 0
        self._is_blinking = False
        self._blink_start_time = None
        self._blink_min_openness = 1.0
        self._last_blink_duration = 0.0
        self._start_time = None

