# -*- coding: utf-8 -*-
"""
Trail system for creating afterimage effects based on gaze movement.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class TrailPoint:
    """Represents a single point in the trail."""
    x: float
    y: float
    radius: float
    alpha: int
    timestamp: float


class TrailSystem:
    """
    Manages the trail/afterimage effect for gaze visualization.
    The trail appearance changes based on velocity and acceleration.
    """
    
    def __init__(
        self,
        max_length: int = 30,
        min_alpha: int = 20,
        max_alpha: int = 200,
        min_radius: float = 5,
        max_radius: float = 18,
        base_color: Tuple[int, int, int] = (255, 50, 50)
    ):
        """
        Initialize the trail system.
        
        Args:
            max_length: Maximum number of trail points to keep
            min_alpha: Minimum alpha for the oldest point
            max_alpha: Maximum alpha for the newest point
            min_radius: Minimum radius for trail circles
            max_radius: Maximum radius for trail circles
            base_color: RGB color for the trail
        """
        self._max_length = max_length
        self._min_alpha = min_alpha
        self._max_alpha = max_alpha
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._base_color = base_color
        
        self._points: deque = deque(maxlen=max_length)
        
        # Velocity thresholds for trail behavior
        self._low_velocity_threshold = 50  # pixels/second
        self._high_velocity_threshold = 500  # pixels/second
    
    def update(
        self,
        x: float,
        y: float,
        velocity_magnitude: float,
        acceleration_magnitude: float,
        timestamp: float
    ) -> None:
        """
        Add a new point to the trail.
        
        Args:
            x: X coordinate of the gaze position
            y: Y coordinate of the gaze position
            velocity_magnitude: Current velocity in pixels/second
            acceleration_magnitude: Current acceleration in pixels/second^2
            timestamp: Time of the measurement
        """
        # Calculate radius based on velocity (inverse relationship)
        # Higher velocity = smaller trail points (more spread out)
        # Lower velocity = larger trail points (more concentrated)
        velocity_factor = self._clamp(
            (velocity_magnitude - self._low_velocity_threshold) / 
            (self._high_velocity_threshold - self._low_velocity_threshold),
            0.0, 1.0
        )
        
        # Inverse relationship: high velocity = small radius
        radius = self._max_radius - velocity_factor * (self._max_radius - self._min_radius)
        
        # Alpha based on acceleration (higher acceleration = more visible trail)
        accel_factor = self._clamp(acceleration_magnitude / 1000.0, 0.0, 1.0)
        base_alpha = int(self._max_alpha * 0.7 + self._max_alpha * 0.3 * accel_factor)
        
        point = TrailPoint(
            x=x,
            y=y,
            radius=radius,
            alpha=base_alpha,
            timestamp=timestamp
        )
        
        self._points.append(point)
    
    def get_trail_points(self) -> List[Tuple[float, float, float, int, Tuple[int, int, int, int]]]:
        """
        Get all trail points with their rendering properties.
        Points are returned from oldest to newest.
        
        Returns:
            List of tuples: (x, y, radius, alpha, (r, g, b, a))
        """
        if not self._points:
            return []
        
        result = []
        num_points = len(self._points)
        
        for i, point in enumerate(self._points):
            # Calculate fade based on position in trail
            # 0 = oldest (most faded), 1 = newest (least faded)
            age_factor = i / max(num_points - 1, 1)
            
            # Interpolate alpha based on age
            alpha = int(self._min_alpha + age_factor * (point.alpha - self._min_alpha))
            
            # Interpolate radius based on age (older points are smaller)
            radius = self._min_radius + age_factor * (point.radius - self._min_radius)
            
            # Color with alpha
            color = (
                self._base_color[0],
                self._base_color[1],
                self._base_color[2],
                alpha
            )
            
            result.append((point.x, point.y, radius, alpha, color))
        
        return result
    
    def get_dynamic_trail_length(self, velocity_magnitude: float) -> int:
        """
        Calculate how many trail points to actually display based on velocity.
        Higher velocity = longer visible trail.
        
        Args:
            velocity_magnitude: Current velocity in pixels/second
            
        Returns:
            Number of trail points to display
        """
        velocity_factor = self._clamp(
            velocity_magnitude / self._high_velocity_threshold,
            0.2, 1.0  # Minimum 20% of trail always visible
        )
        
        return max(5, int(self._max_length * velocity_factor))
    
    def clear(self) -> None:
        """Clear all trail points."""
        self._points.clear()
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max."""
        return max(min_val, min(max_val, value))
    
    @property
    def point_count(self) -> int:
        """Get the current number of trail points."""
        return len(self._points)
    
    @property
    def base_color(self) -> Tuple[int, int, int]:
        """Get the base color of the trail."""
        return self._base_color
    
    @base_color.setter
    def base_color(self, color: Tuple[int, int, int]) -> None:
        """Set the base color of the trail."""
        self._base_color = color

