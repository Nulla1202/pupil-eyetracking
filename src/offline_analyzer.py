# -*- coding: utf-8 -*-
"""
Offline Analyzer - Process video files for pupil measurements.
Analyzes recorded videos and exports metrics to CSV/JSON.
"""

import cv2
import json
import time
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from .pupil_detector import PupilDetector, PupilData
from .metrics import MetricsCalculator, PupilMetrics


class OfflineAnalyzer:
    """
    Analyzes video files for pupil metrics offline.
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.2,
        show_progress: bool = True
    ):
        """
        Initialize the offline analyzer.
        
        Args:
            ear_threshold: EAR threshold for blink detection
            show_progress: Whether to show progress bar
        """
        self._ear_threshold = ear_threshold
        self._show_progress = show_progress
        
        self._pupil_detector: Optional[PupilDetector] = None
        self._metrics_calc: Optional[MetricsCalculator] = None
        
        # Results storage
        self._frame_data: List[Dict[str, Any]] = []
        self._summary: Dict[str, Any] = {}
    
    def analyze(self, video_path: str) -> bool:
        """
        Analyze a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if analysis was successful
        """
        path = Path(video_path)
        if not path.exists():
            print(f"Error: Video file not found: {video_path}")
            return False
        
        # Open video
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {duration:.2f}s ({total_frames} frames)")
        print()
        
        # Initialize detectors
        self._pupil_detector = PupilDetector()
        self._metrics_calc = MetricsCalculator(
            ear_threshold=self._ear_threshold
        )
        
        # Clear previous results
        self._frame_data = []
        
        # Process frames
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp based on frame number
                timestamp = frame_idx / fps if fps > 0 else 0
                
                # Process frame
                pupil_data = self._pupil_detector.process_frame(frame, timestamp)
                metrics = self._metrics_calc.update(pupil_data)
                
                # Store frame data
                self._frame_data.append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'left_pupil_x': pupil_data.left_pupil_x,
                    'left_pupil_y': pupil_data.left_pupil_y,
                    'left_diameter_px': pupil_data.left_diameter_px,
                    'left_diameter_mm': pupil_data.left_diameter_mm,
                    'right_pupil_x': pupil_data.right_pupil_x,
                    'right_pupil_y': pupil_data.right_pupil_y,
                    'right_diameter_px': pupil_data.right_diameter_px,
                    'right_diameter_mm': pupil_data.right_diameter_mm,
                    'left_ear': pupil_data.left_ear,
                    'right_ear': pupil_data.right_ear,
                    'avg_ear': pupil_data.avg_ear,
                    'is_blinking': metrics.is_blinking,
                    'is_valid': pupil_data.is_valid
                })
                
                # Show progress
                if self._show_progress:
                    self._print_progress(frame_idx + 1, total_frames)
                
                frame_idx += 1
        
        finally:
            cap.release()
            if self._pupil_detector:
                self._pupil_detector.release()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate summary
        self._generate_summary(
            video_path=str(path),
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            processing_time=processing_time
        )
        
        if self._show_progress:
            print()  # New line after progress bar
        
        print(f"\nProcessing complete!")
        print(f"  Frames processed: {frame_idx}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Processing speed: {frame_idx/processing_time:.1f} fps")
        
        return True
    
    def _print_progress(self, current: int, total: int) -> None:
        """Print a progress bar."""
        bar_length = 40
        progress = current / total
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        percent = progress * 100
        sys.stdout.write(f'\rProgress: [{bar}] {percent:.1f}% ({current}/{total})')
        sys.stdout.flush()
    
    def _generate_summary(
        self,
        video_path: str,
        total_frames: int,
        fps: float,
        duration: float,
        processing_time: float
    ) -> None:
        """Generate analysis summary statistics."""
        if not self._frame_data or not self._metrics_calc:
            return
        
        # Get valid frames
        valid_frames = [f for f in self._frame_data if f['is_valid']]
        
        if not valid_frames:
            self._summary = {
                'video_path': video_path,
                'total_frames': total_frames,
                'valid_frames': 0,
                'error': 'No valid detections'
            }
            return
        
        # Calculate statistics
        import numpy as np
        
        left_diameters = [f['left_diameter_mm'] for f in valid_frames]
        right_diameters = [f['right_diameter_mm'] for f in valid_frames]
        ears = [f['avg_ear'] for f in valid_frames]
        
        # Get blink data
        blink_events = self._metrics_calc.get_blink_events()
        
        self._summary = {
            'video_info': {
                'path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'duration_seconds': duration
            },
            'processing': {
                'processing_time_seconds': processing_time,
                'processing_fps': total_frames / processing_time if processing_time > 0 else 0
            },
            'detection': {
                'valid_frames': len(valid_frames),
                'invalid_frames': total_frames - len(valid_frames),
                'detection_rate': len(valid_frames) / total_frames * 100 if total_frames > 0 else 0
            },
            'pupil_diameter': {
                'left': {
                    'mean_mm': float(np.mean(left_diameters)),
                    'std_mm': float(np.std(left_diameters)),
                    'min_mm': float(np.min(left_diameters)),
                    'max_mm': float(np.max(left_diameters))
                },
                'right': {
                    'mean_mm': float(np.mean(right_diameters)),
                    'std_mm': float(np.std(right_diameters)),
                    'min_mm': float(np.min(right_diameters)),
                    'max_mm': float(np.max(right_diameters))
                }
            },
            'ear': {
                'mean': float(np.mean(ears)),
                'std': float(np.std(ears)),
                'min': float(np.min(ears)),
                'max': float(np.max(ears))
            },
            'blinks': {
                'total_count': len(blink_events),
                'blinks_per_minute': len(blink_events) / (duration / 60) if duration > 0 else 0,
                'avg_duration_ms': float(np.mean([b.duration * 1000 for b in blink_events])) if blink_events else 0,
                'events': [
                    {
                        'start_time': b.start_time,
                        'end_time': b.end_time,
                        'duration_ms': b.duration * 1000,
                        'min_ear': b.min_ear
                    }
                    for b in blink_events
                ]
            }
        }
    
    def save_csv(self, output_path: str) -> bool:
        """
        Save frame-by-frame data to CSV.
        
        Args:
            output_path: Path for the output CSV file
            
        Returns:
            True if successful
        """
        if not self._frame_data:
            print("Error: No data to save. Run analyze() first.")
            return False
        
        try:
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if self._frame_data:
                    writer = csv.DictWriter(f, fieldnames=self._frame_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self._frame_data)
            
            print(f"CSV saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    def save_json(self, output_path: str, include_frames: bool = False) -> bool:
        """
        Save analysis results to JSON.
        
        Args:
            output_path: Path for the output JSON file
            include_frames: Whether to include frame-by-frame data
            
        Returns:
            True if successful
        """
        if not self._summary:
            print("Error: No data to save. Run analyze() first.")
            return False
        
        try:
            data = self._summary.copy()
            
            if include_frames:
                data['frames'] = self._frame_data
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"JSON saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get the analysis summary."""
        return self._summary.copy()
    
    def get_frame_data(self) -> List[Dict[str, Any]]:
        """Get frame-by-frame data."""
        return self._frame_data.copy()
    
    def print_summary(self) -> None:
        """Print a formatted summary of the analysis."""
        if not self._summary:
            print("No analysis data available.")
            return
        
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        
        if 'video_info' in self._summary:
            vi = self._summary['video_info']
            print(f"\nVideo: {vi['path']}")
            print(f"Duration: {vi['duration_seconds']:.2f}s")
        
        if 'detection' in self._summary:
            det = self._summary['detection']
            print(f"\nDetection Rate: {det['detection_rate']:.1f}%")
            print(f"Valid Frames: {det['valid_frames']}/{det['valid_frames'] + det['invalid_frames']}")
        
        if 'pupil_diameter' in self._summary:
            pd = self._summary['pupil_diameter']
            print(f"\nPupil Diameter (mm):")
            print(f"  Left:  {pd['left']['mean_mm']:.2f} ± {pd['left']['std_mm']:.2f}")
            print(f"  Right: {pd['right']['mean_mm']:.2f} ± {pd['right']['std_mm']:.2f}")
        
        if 'ear' in self._summary:
            ear = self._summary['ear']
            print(f"\nEye Aspect Ratio:")
            print(f"  Mean: {ear['mean']:.3f} ± {ear['std']:.3f}")
        
        if 'blinks' in self._summary:
            bl = self._summary['blinks']
            print(f"\nBlinks:")
            print(f"  Total: {bl['total_count']}")
            print(f"  Rate: {bl['blinks_per_minute']:.1f} per minute")
            if bl['avg_duration_ms'] > 0:
                print(f"  Avg Duration: {bl['avg_duration_ms']:.0f} ms")
        
        print("=" * 50)


def run_offline_analysis(
    video_path: str,
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
    ear_threshold: float = 0.2
) -> bool:
    """
    Convenience function to run offline analysis.
    
    Args:
        video_path: Path to video file
        output_csv: Optional path for CSV output
        output_json: Optional path for JSON output
        ear_threshold: EAR threshold for blink detection
        
    Returns:
        True if successful
    """
    analyzer = OfflineAnalyzer(ear_threshold=ear_threshold)
    
    if not analyzer.analyze(video_path):
        return False
    
    # Auto-generate output paths if not provided
    video_stem = Path(video_path).stem
    
    if output_csv is None:
        output_csv = f"{video_stem}_pupil_data.csv"
    
    if output_json is None:
        output_json = f"{video_stem}_pupil_summary.json"
    
    # Save results
    analyzer.save_csv(output_csv)
    analyzer.save_json(output_json)
    
    # Print summary
    analyzer.print_summary()
    
    return True

