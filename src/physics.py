# -*- coding: utf-8 -*-
"""
Physics module for calculating velocity and acceleration from gaze data.
"""

import time
import math
from collections import deque
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class GazeState:
    """Represents the current state of gaze tracking with physics data."""
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    velocity_magnitude: float
    acceleration_x: float
    acceleration_y: float
    acceleration_magnitude: float
    timestamp: float


class PhysicsCalculator:
    """
    Calculates velocity and acceleration from gaze position data.
    Uses smoothing to reduce noise in the measurements.
    """
    
    def __init__(self, smoothing_factor: float = 0.3, history_size: int = 10):
        """
        Initialize the physics calculator.
        
        Args:
            smoothing_factor: Low-pass filter factor (0-1, lower = more smoothing)
            history_size: Number of samples to keep for calculations
        """
        self._smoothing_factor = smoothing_factor
        self._history_size = history_size
        
        # Position history: (x, y, timestamp)
        self._position_history: deque = deque(maxlen=history_size)
        
        # Smoothed values
        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None
        self._smoothed_vx: float = 0.0
        self._smoothed_vy: float = 0.0
        self._smoothed_ax: float = 0.0
        self._smoothed_ay: float = 0.0
        
        # Previous velocity for acceleration calculation
        self._prev_vx: float = 0.0
        self._prev_vy: float = 0.0
        self._prev_velocity_time: float = 0.0
    
    def update(self, x: float, y: float, timestamp: Optional[float] = None) -> GazeState:
        """
        Update with a new gaze position and calculate physics.
        
        Args:
            x: X coordinate of gaze position
            y: Y coordinate of gaze position
            timestamp: Time of the measurement (uses current time if None)
            
        Returns:
            GazeState with all calculated values
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Apply low-pass filter for smoothing
        if self._smoothed_x is None:
            self._smoothed_x = x
            self._smoothed_y = y
        else:
            self._smoothed_x = (self._smoothing_factor * x + 
                               (1 - self._smoothing_factor) * self._smoothed_x)
            self._smoothed_y = (self._smoothing_factor * y + 
                               (1 - self._smoothing_factor) * self._smoothed_y)
        
        # Store in history
        self._position_history.append((self._smoothed_x, self._smoothed_y, timestamp))
        
        # Calculate velocity
        vx, vy = self._calculate_velocity()
        
        # Apply smoothing to velocity
        self._smoothed_vx = (self._smoothing_factor * vx + 
                            (1 - self._smoothing_factor) * self._smoothed_vx)
        self._smoothed_vy = (self._smoothing_factor * vy + 
                            (1 - self._smoothing_factor) * self._smoothed_vy)
        
        # Calculate acceleration
        ax, ay = self._calculate_acceleration(timestamp)
        
        # Apply smoothing to acceleration
        self._smoothed_ax = (self._smoothing_factor * ax + 
                            (1 - self._smoothing_factor) * self._smoothed_ax)
        self._smoothed_ay = (self._smoothing_factor * ay + 
                            (1 - self._smoothing_factor) * self._smoothed_ay)
        
        # Update previous velocity for next acceleration calculation
        self._prev_vx = self._smoothed_vx
        self._prev_vy = self._smoothed_vy
        self._prev_velocity_time = timestamp
        
        # Calculate magnitudes
        velocity_mag = math.sqrt(self._smoothed_vx**2 + self._smoothed_vy**2)
        accel_mag = math.sqrt(self._smoothed_ax**2 + self._smoothed_ay**2)
        
        return GazeState(
            x=self._smoothed_x,
            y=self._smoothed_y,
            velocity_x=self._smoothed_vx,
            velocity_y=self._smoothed_vy,
            velocity_magnitude=velocity_mag,
            acceleration_x=self._smoothed_ax,
            acceleration_y=self._smoothed_ay,
            acceleration_magnitude=accel_mag,
            timestamp=timestamp
        )
    
    def _calculate_velocity(self) -> Tuple[float, float]:
        """
        Calculate velocity from position history.
        
        Returns:
            Tuple of (velocity_x, velocity_y) in pixels per second
        """
        if len(self._position_history) < 2:
            return (0.0, 0.0)
        
        # Use the two most recent positions
        current = self._position_history[-1]
        previous = self._position_history[-2]
        
        dt = current[2] - previous[2]
        if dt <= 0:
            return (self._smoothed_vx, self._smoothed_vy)
        
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        
        vx = dx / dt
        vy = dy / dt
        
        return (vx, vy)
    
    def _calculate_acceleration(self, current_time: float) -> Tuple[float, float]:
        """
        Calculate acceleration from velocity change.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Tuple of (acceleration_x, acceleration_y) in pixels per second squared
        """
        dt = current_time - self._prev_velocity_time
        if dt <= 0:
            return (self._smoothed_ax, self._smoothed_ay)
        
        dvx = self._smoothed_vx - self._prev_vx
        dvy = self._smoothed_vy - self._prev_vy
        
        ax = dvx / dt
        ay = dvy / dt
        
        return (ax, ay)
    
    def reset(self) -> None:
        """Reset all calculations and history."""
        self._position_history.clear()
        self._smoothed_x = None
        self._smoothed_y = None
        self._smoothed_vx = 0.0
        self._smoothed_vy = 0.0
        self._smoothed_ax = 0.0
        self._smoothed_ay = 0.0
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_velocity_time = 0.0
    
    def get_position_history(self) -> list:
        """
        Get the position history.
        
        Returns:
            List of (x, y, timestamp) tuples
        """
        return list(self._position_history)

