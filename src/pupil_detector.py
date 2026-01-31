# -*- coding: utf-8 -*-
"""
PupilDetector - Real-time pupil detection using MediaPipe Face Mesh.
Detects iris landmarks and calculates pupil diameter and position.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class PupilData:
    """Data class holding pupil measurement results."""
    # Left eye
    left_pupil_x: float
    left_pupil_y: float
    left_diameter_px: float
    left_diameter_mm: float
    
    # Right eye
    right_pupil_x: float
    right_pupil_y: float
    right_diameter_px: float
    right_diameter_mm: float
    
    # Eye Aspect Ratio
    left_ear: float
    right_ear: float
    avg_ear: float
    
    # Detection status
    is_valid: bool
    timestamp: float


class PupilDetector:
    """
    Detects pupils using MediaPipe Face Mesh iris landmarks.
    
    MediaPipe Face Mesh provides 478 landmarks, with iris landmarks at indices:
    - Left iris: 468, 469, 470, 471, 472 (center at 468)
    - Right iris: 473, 474, 475, 476, 477 (center at 473)
    
    Eye landmarks for EAR calculation:
    - Left eye: 33, 160, 158, 133, 153, 144
    - Right eye: 362, 385, 387, 263, 373, 380
    """
    
    # Iris landmark indices
    LEFT_IRIS_CENTER = 468
    LEFT_IRIS_POINTS = [468, 469, 470, 471]
    RIGHT_IRIS_CENTER = 473
    RIGHT_IRIS_POINTS = [473, 474, 475, 476]
    
    # Eye landmarks for EAR calculation (vertical and horizontal points)
    # Left eye: P1=33, P2=160, P3=158, P4=133, P5=153, P6=144
    LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    # Right eye: P1=362, P2=385, P3=387, P4=263, P5=373, P6=380
    RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
    
    # Average iris diameter in mm (for calibration)
    AVERAGE_IRIS_DIAMETER_MM = 11.7
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True
    ):
        """
        Initialize the PupilDetector.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            refine_landmarks: Whether to refine landmarks around eyes and lips
        """
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=refine_landmarks,  # Required for iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self._last_valid_data: Optional[PupilData] = None
        self._pixels_per_mm: float = 1.0  # Will be calibrated
        self._is_calibrated: bool = False
    
    def process_frame(self, frame: np.ndarray, timestamp: float = 0.0) -> PupilData:
        """
        Process a frame and detect pupil data.
        
        Args:
            frame: BGR image from camera
            timestamp: Timestamp of the frame
            
        Returns:
            PupilData with detection results
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Process with MediaPipe
        results = self._face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return self._create_invalid_data(timestamp)
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get iris data
        left_iris = self._get_iris_data(landmarks, self.LEFT_IRIS_CENTER, self.LEFT_IRIS_POINTS, w, h)
        right_iris = self._get_iris_data(landmarks, self.RIGHT_IRIS_CENTER, self.RIGHT_IRIS_POINTS, w, h)
        
        # Get EAR values
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_POINTS, w, h)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_POINTS, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Auto-calibrate using iris size if not yet calibrated
        if not self._is_calibrated and left_iris[2] > 0 and right_iris[2] > 0:
            avg_iris_px = (left_iris[2] + right_iris[2]) / 2.0
            self._pixels_per_mm = avg_iris_px / self.AVERAGE_IRIS_DIAMETER_MM
            self._is_calibrated = True
        
        # Convert pixel diameter to mm
        left_diameter_mm = left_iris[2] / self._pixels_per_mm if self._pixels_per_mm > 0 else 0
        right_diameter_mm = right_iris[2] / self._pixels_per_mm if self._pixels_per_mm > 0 else 0
        
        data = PupilData(
            left_pupil_x=left_iris[0],
            left_pupil_y=left_iris[1],
            left_diameter_px=left_iris[2],
            left_diameter_mm=left_diameter_mm,
            right_pupil_x=right_iris[0],
            right_pupil_y=right_iris[1],
            right_diameter_px=right_iris[2],
            right_diameter_mm=right_diameter_mm,
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            is_valid=True,
            timestamp=timestamp
        )
        
        self._last_valid_data = data
        return data
    
    def _get_iris_data(
        self,
        landmarks,
        center_idx: int,
        iris_indices: list,
        width: int,
        height: int
    ) -> Tuple[float, float, float]:
        """
        Extract iris center and diameter from landmarks.
        
        Returns:
            Tuple of (center_x, center_y, diameter_px)
        """
        # Get center point
        center = landmarks[center_idx]
        center_x = center.x * width
        center_y = center.y * height
        
        # Calculate diameter from iris points
        iris_points = []
        for idx in iris_indices:
            lm = landmarks[idx]
            iris_points.append((lm.x * width, lm.y * height))
        
        # Calculate diameter as max distance between opposite points
        if len(iris_points) >= 4:
            # Horizontal diameter (points 1 and 3 are typically horizontal)
            h_dist = np.sqrt(
                (iris_points[1][0] - iris_points[3][0])**2 +
                (iris_points[1][1] - iris_points[3][1])**2
            )
            # Vertical diameter (points 0 and 2 are typically vertical)
            v_dist = np.sqrt(
                (iris_points[0][0] - iris_points[2][0])**2 +
                (iris_points[0][1] - iris_points[2][1])**2
            )
            diameter = (h_dist + v_dist) / 2.0
        else:
            diameter = 0.0
        
        return (center_x, center_y, diameter)
    
    def _calculate_ear(
        self,
        landmarks,
        eye_points: list,
        width: int,
        height: int
    ) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        
        Where P1-P6 are the eye landmarks:
        P1, P4: horizontal corners
        P2, P3: upper lid
        P5, P6: lower lid
        """
        # Extract points
        points = []
        for idx in eye_points:
            lm = landmarks[idx]
            points.append((lm.x * width, lm.y * height))
        
        # P1=0, P2=1, P3=2, P4=3, P5=4, P6=5
        # Vertical distances
        v1 = np.sqrt((points[1][0] - points[5][0])**2 + (points[1][1] - points[5][1])**2)
        v2 = np.sqrt((points[2][0] - points[4][0])**2 + (points[2][1] - points[4][1])**2)
        
        # Horizontal distance
        h = np.sqrt((points[0][0] - points[3][0])**2 + (points[0][1] - points[3][1])**2)
        
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _create_invalid_data(self, timestamp: float) -> PupilData:
        """Create a PupilData object for invalid/missing detection."""
        if self._last_valid_data is not None:
            # Return last valid data with is_valid=False
            return PupilData(
                left_pupil_x=self._last_valid_data.left_pupil_x,
                left_pupil_y=self._last_valid_data.left_pupil_y,
                left_diameter_px=self._last_valid_data.left_diameter_px,
                left_diameter_mm=self._last_valid_data.left_diameter_mm,
                right_pupil_x=self._last_valid_data.right_pupil_x,
                right_pupil_y=self._last_valid_data.right_pupil_y,
                right_diameter_px=self._last_valid_data.right_diameter_px,
                right_diameter_mm=self._last_valid_data.right_diameter_mm,
                left_ear=self._last_valid_data.left_ear,
                right_ear=self._last_valid_data.right_ear,
                avg_ear=self._last_valid_data.avg_ear,
                is_valid=False,
                timestamp=timestamp
            )
        
        return PupilData(
            left_pupil_x=0, left_pupil_y=0,
            left_diameter_px=0, left_diameter_mm=0,
            right_pupil_x=0, right_pupil_y=0,
            right_diameter_px=0, right_diameter_mm=0,
            left_ear=0, right_ear=0, avg_ear=0,
            is_valid=False,
            timestamp=timestamp
        )
    
    def release(self) -> None:
        """Release resources."""
        if self._face_mesh:
            self._face_mesh.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

