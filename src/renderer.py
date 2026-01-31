# -*- coding: utf-8 -*-
"""
Pygame renderer for gaze tracking visualization.
"""

import pygame
from typing import Tuple, List, Optional
import os

from .physics import GazeState
from .trail import TrailSystem
from .metrics import PupilMetrics


class Renderer:
    """
    Handles all Pygame rendering for the gaze tracking visualization.
    """
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fullscreen: bool = True,
        background_color: Tuple[int, int, int] = (15, 15, 25),
        gaze_color: Tuple[int, int, int] = (255, 50, 50),
        gaze_radius: int = 20
    ):
        """
        Initialize the renderer.
        
        Args:
            width: Screen width
            height: Screen height
            fullscreen: Whether to use fullscreen mode
            background_color: RGB background color
            gaze_color: RGB color for the gaze circle
            gaze_radius: Radius of the main gaze circle
        """
        self._width = width
        self._height = height
        self._fullscreen = fullscreen
        self._background_color = background_color
        self._gaze_color = gaze_color
        self._gaze_radius = gaze_radius
        
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._title_font: Optional[pygame.font.Font] = None
        
        # Info panel settings (right side - gaze)
        self._panel_padding = 20
        self._panel_width = 280
        self._panel_height = 160
        self._panel_bg_color = (30, 30, 30)
        self._panel_text_color = (220, 220, 220)
        self._panel_accent_color = (255, 100, 100)
        
        # Pupil panel settings (left side)
        self._pupil_panel_width = 320
        self._pupil_panel_height = 280
        self._pupil_panel_bg_color = (30, 30, 40)
        self._pupil_panel_accent_color = (100, 200, 255)  # Cyan
        
        # EAR graph settings
        self._ear_graph_height = 60
        self._ear_graph_samples = 100
        self._ear_history: List[float] = []
        self._ear_threshold = 0.2
        
        # Blink indicator
        self._blink_flash_frames = 0
        self._blink_flash_duration = 10  # frames
        
        self._is_initialized = False
        self._current_fps = 0.0
    
    def initialize(self) -> bool:
        """
        Initialize Pygame and create the display.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            pygame.init()
            pygame.font.init()
            
            # Get display info for fullscreen
            display_info = pygame.display.Info()
            if self._fullscreen:
                self._width = display_info.current_w
                self._height = display_info.current_h
                flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            else:
                flags = pygame.HWSURFACE | pygame.DOUBLEBUF
            
            self._screen = pygame.display.set_mode(
                (self._width, self._height),
                flags
            )
            pygame.display.set_caption("GazeTrack - Eye Tracking Visualization")
            
            # Hide mouse cursor
            pygame.mouse.set_visible(False)
            
            # Initialize clock
            self._clock = pygame.time.Clock()
            
            # Initialize fonts
            self._font = pygame.font.SysFont('Consolas', 22)
            self._small_font = pygame.font.SysFont('Consolas', 18)
            self._title_font = pygame.font.SysFont('Consolas', 20, bold=True)
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize renderer: {e}")
            return False
    
    def render(
        self,
        gaze_state: GazeState,
        trail_system: TrailSystem,
        target_fps: int = 60,
        pupil_metrics: Optional[PupilMetrics] = None,
        ear_history: Optional[List[float]] = None
    ) -> float:
        """
        Render a frame.
        
        Args:
            gaze_state: Current gaze state with position and physics
            trail_system: Trail system for afterimage effect
            target_fps: Target frames per second
            pupil_metrics: Optional pupil measurement data
            ear_history: Optional EAR history for graph
            
        Returns:
            float: Actual FPS achieved
        """
        if not self._is_initialized or self._screen is None:
            return 0.0
        
        # Clear screen
        self._screen.fill(self._background_color)
        
        # Draw blink flash overlay if blinking
        if pupil_metrics and pupil_metrics.is_blinking:
            self._blink_flash_frames = self._blink_flash_duration
        
        if self._blink_flash_frames > 0:
            self._draw_blink_indicator()
            self._blink_flash_frames -= 1
        
        # Draw trail
        self._draw_trail(trail_system, gaze_state.velocity_magnitude)
        
        # Draw main gaze circle
        self._draw_gaze_circle(gaze_state.x, gaze_state.y)
        
        # Draw glow effect around main circle
        self._draw_gaze_glow(gaze_state.x, gaze_state.y)
        
        # Draw info panel (right side)
        self._draw_info_panel(gaze_state)
        
        # Draw pupil panel if metrics available (left side)
        if pupil_metrics is not None:
            self._draw_pupil_panel(pupil_metrics, ear_history)
        
        # Update display
        pygame.display.flip()
        
        # Tick clock
        if self._clock is not None:
            self._clock.tick(target_fps)
            self._current_fps = self._clock.get_fps()
        
        return self._current_fps
    
    def render_pupil_only(
        self,
        target_fps: int = 60,
        pupil_metrics: Optional[PupilMetrics] = None,
        ear_history: Optional[List[float]] = None
    ) -> float:
        """
        Render pupil-only mode (no gaze tracking).
        
        Args:
            target_fps: Target frames per second
            pupil_metrics: Pupil measurement data
            ear_history: EAR history for graph
            
        Returns:
            float: Actual FPS achieved
        """
        if not self._is_initialized or self._screen is None:
            return 0.0
        
        # Clear screen
        self._screen.fill(self._background_color)
        
        # Draw blink flash overlay if blinking
        if pupil_metrics and pupil_metrics.is_blinking:
            self._blink_flash_frames = self._blink_flash_duration
        
        if self._blink_flash_frames > 0:
            self._draw_blink_indicator()
            self._blink_flash_frames -= 1
        
        # Draw centered pupil visualization
        if pupil_metrics is not None:
            self._draw_pupil_visualization(pupil_metrics)
            self._draw_pupil_panel_centered(pupil_metrics, ear_history)
        
        # Update display
        pygame.display.flip()
        
        # Tick clock
        if self._clock is not None:
            self._clock.tick(target_fps)
            self._current_fps = self._clock.get_fps()
        
        return self._current_fps
    
    def _draw_blink_indicator(self) -> None:
        """Draw a flash effect when blinking is detected."""
        if self._screen is None:
            return
        
        # Create semi-transparent overlay
        alpha = int(100 * (self._blink_flash_frames / self._blink_flash_duration))
        overlay = pygame.Surface((self._width, self._height), pygame.SRCALPHA)
        overlay.fill((255, 255, 100, alpha))  # Yellow flash
        self._screen.blit(overlay, (0, 0))
    
    def _draw_pupil_panel(
        self,
        metrics: PupilMetrics,
        ear_history: Optional[List[float]] = None
    ) -> None:
        """Draw the pupil metrics panel on the left side."""
        if self._screen is None or self._font is None:
            return
        
        # Panel position (top-left)
        panel_x = self._panel_padding
        panel_y = self._panel_padding
        
        # Draw panel background
        panel_surface = pygame.Surface(
            (self._pupil_panel_width, self._pupil_panel_height),
            pygame.SRCALPHA
        )
        
        pygame.draw.rect(
            panel_surface,
            (*self._pupil_panel_bg_color, 220),
            (0, 0, self._pupil_panel_width, self._pupil_panel_height),
            border_radius=10
        )
        
        pygame.draw.rect(
            panel_surface,
            (*self._pupil_panel_accent_color, 150),
            (0, 0, self._pupil_panel_width, self._pupil_panel_height),
            width=2,
            border_radius=10
        )
        
        self._screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw content
        line_height = 24
        text_x = panel_x + 15
        text_y = panel_y + 12
        
        # Title
        title = self._title_font.render("PUPIL METRICS", True, self._pupil_panel_accent_color)
        self._screen.blit(title, (text_x, text_y))
        text_y += line_height + 8
        
        # Pupil diameters (show N/A if not available)
        left_color = (100, 150, 255)  # Blue
        right_color = (255, 150, 100)  # Orange
        
        if metrics.current_left_diameter_mm > 0:
            left_text = f"Left Pupil:  {metrics.current_left_diameter_mm:.2f} mm"
            left_surface = self._font.render(left_text, True, left_color)
            self._screen.blit(left_surface, (text_x, text_y))
            text_y += line_height
            
            right_text = f"Right Pupil: {metrics.current_right_diameter_mm:.2f} mm"
            right_surface = self._font.render(right_text, True, right_color)
            self._screen.blit(right_surface, (text_x, text_y))
            text_y += line_height
            
            # Average
            avg_text = f"Average:     {metrics.current_avg_diameter_mm:.2f} mm"
            avg_surface = self._font.render(avg_text, True, self._panel_text_color)
            self._screen.blit(avg_surface, (text_x, text_y))
            text_y += line_height + 5
        else:
            # Pupil diameter not available (gaze+pupil mode uses eye openness only)
            note_text = "Eye Openness Mode"
            note_surface = self._font.render(note_text, True, (150, 150, 150))
            self._screen.blit(note_surface, (text_x, text_y))
            text_y += line_height + 5
        
        # EAR
        ear_color = (100, 255, 150) if metrics.current_ear > self._ear_threshold else (255, 100, 100)
        ear_text = f"EAR: {metrics.current_ear:.3f}"
        ear_surface = self._font.render(ear_text, True, ear_color)
        self._screen.blit(ear_surface, (text_x, text_y))
        text_y += line_height
        
        # Blink info
        blink_text = f"Blinks: {metrics.blink_count} ({metrics.blinks_per_minute:.1f}/min)"
        blink_surface = self._font.render(blink_text, True, self._panel_text_color)
        self._screen.blit(blink_surface, (text_x, text_y))
        text_y += line_height + 8
        
        # Draw EAR graph
        if ear_history:
            graph_x = panel_x + 15
            graph_y = text_y
            graph_width = self._pupil_panel_width - 30
            graph_height = self._ear_graph_height
            
            self._draw_ear_graph(ear_history, graph_x, graph_y, graph_width, graph_height)
    
    def _draw_pupil_panel_centered(
        self,
        metrics: PupilMetrics,
        ear_history: Optional[List[float]] = None
    ) -> None:
        """Draw a larger centered pupil panel for pupil-only mode."""
        if self._screen is None or self._font is None:
            return
        
        # Centered panel
        panel_width = 400
        panel_height = 350
        panel_x = (self._width - panel_width) // 2
        panel_y = self._height - panel_height - 50
        
        # Draw panel background
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        
        pygame.draw.rect(
            panel_surface,
            (*self._pupil_panel_bg_color, 230),
            (0, 0, panel_width, panel_height),
            border_radius=15
        )
        
        pygame.draw.rect(
            panel_surface,
            (*self._pupil_panel_accent_color, 180),
            (0, 0, panel_width, panel_height),
            width=2,
            border_radius=15
        )
        
        self._screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw content
        line_height = 28
        text_x = panel_x + 20
        text_y = panel_y + 15
        
        # Title
        title = self._title_font.render("PUPIL MEASUREMENT", True, self._pupil_panel_accent_color)
        self._screen.blit(title, (text_x, text_y))
        text_y += line_height + 10
        
        # Large pupil diameter display
        left_color = (100, 150, 255)
        right_color = (255, 150, 100)
        
        # Left eye
        left_label = self._font.render("Left Eye:", True, left_color)
        self._screen.blit(left_label, (text_x, text_y))
        left_value = self._title_font.render(f"{metrics.current_left_diameter_mm:.2f} mm", True, left_color)
        self._screen.blit(left_value, (text_x + 150, text_y))
        text_y += line_height
        
        # Right eye
        right_label = self._font.render("Right Eye:", True, right_color)
        self._screen.blit(right_label, (text_x, text_y))
        right_value = self._title_font.render(f"{metrics.current_right_diameter_mm:.2f} mm", True, right_color)
        self._screen.blit(right_value, (text_x + 150, text_y))
        text_y += line_height + 10
        
        # EAR
        ear_color = (100, 255, 150) if metrics.current_ear > self._ear_threshold else (255, 100, 100)
        ear_label = self._font.render("Eye Aspect Ratio:", True, self._panel_text_color)
        self._screen.blit(ear_label, (text_x, text_y))
        ear_value = self._title_font.render(f"{metrics.current_ear:.3f}", True, ear_color)
        self._screen.blit(ear_value, (text_x + 200, text_y))
        text_y += line_height
        
        # Blink stats
        blink_label = self._font.render("Blink Count:", True, self._panel_text_color)
        self._screen.blit(blink_label, (text_x, text_y))
        blink_value = self._title_font.render(f"{metrics.blink_count}", True, self._panel_text_color)
        self._screen.blit(blink_value, (text_x + 200, text_y))
        text_y += line_height
        
        rate_label = self._font.render("Blink Rate:", True, self._panel_text_color)
        self._screen.blit(rate_label, (text_x, text_y))
        rate_value = self._title_font.render(f"{metrics.blinks_per_minute:.1f}/min", True, self._panel_text_color)
        self._screen.blit(rate_value, (text_x + 200, text_y))
        text_y += line_height + 10
        
        # EAR Graph
        if ear_history:
            graph_x = panel_x + 20
            graph_y = text_y
            graph_width = panel_width - 40
            graph_height = 70
            
            self._draw_ear_graph(ear_history, graph_x, graph_y, graph_width, graph_height)
    
    def _draw_pupil_visualization(self, metrics: PupilMetrics) -> None:
        """Draw visual representation of pupils in the center."""
        if self._screen is None:
            return
        
        # Center position for eye visualization
        center_y = self._height // 3
        left_center_x = self._width // 2 - 150
        right_center_x = self._width // 2 + 150
        
        # Scale factor for visualization (1mm = 8 pixels)
        scale = 8
        
        # Draw eye outlines (white circles)
        eye_radius = 80
        pygame.draw.circle(self._screen, (60, 60, 70), (left_center_x, center_y), eye_radius)
        pygame.draw.circle(self._screen, (60, 60, 70), (right_center_x, center_y), eye_radius)
        pygame.draw.circle(self._screen, (100, 100, 110), (left_center_x, center_y), eye_radius, 2)
        pygame.draw.circle(self._screen, (100, 100, 110), (right_center_x, center_y), eye_radius, 2)
        
        # Draw iris (colored ring)
        iris_radius = 50
        iris_color_left = (80, 120, 180)
        iris_color_right = (180, 120, 80)
        pygame.draw.circle(self._screen, iris_color_left, (left_center_x, center_y), iris_radius)
        pygame.draw.circle(self._screen, iris_color_right, (right_center_x, center_y), iris_radius)
        
        # Draw pupils (black circles with size based on measurement)
        left_pupil_radius = int(metrics.current_left_diameter_mm * scale / 2)
        right_pupil_radius = int(metrics.current_right_diameter_mm * scale / 2)
        
        # Clamp to reasonable sizes
        left_pupil_radius = max(5, min(45, left_pupil_radius))
        right_pupil_radius = max(5, min(45, right_pupil_radius))
        
        pygame.draw.circle(self._screen, (10, 10, 15), (left_center_x, center_y), left_pupil_radius)
        pygame.draw.circle(self._screen, (10, 10, 15), (right_center_x, center_y), right_pupil_radius)
        
        # Add highlight
        highlight_offset = left_pupil_radius // 3
        pygame.draw.circle(
            self._screen, (255, 255, 255),
            (left_center_x - highlight_offset, center_y - highlight_offset),
            max(2, left_pupil_radius // 4)
        )
        pygame.draw.circle(
            self._screen, (255, 255, 255),
            (right_center_x - highlight_offset, center_y - highlight_offset),
            max(2, right_pupil_radius // 4)
        )
        
        # Labels
        if self._small_font:
            left_label = self._small_font.render("LEFT", True, (100, 150, 255))
            right_label = self._small_font.render("RIGHT", True, (255, 150, 100))
            self._screen.blit(left_label, (left_center_x - 25, center_y + eye_radius + 10))
            self._screen.blit(right_label, (right_center_x - 30, center_y + eye_radius + 10))
    
    def _draw_ear_graph(
        self,
        ear_history: List[float],
        x: int,
        y: int,
        width: int,
        height: int
    ) -> None:
        """Draw the EAR history graph."""
        if self._screen is None or not ear_history:
            return
        
        # Draw background
        graph_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(graph_surface, (20, 20, 30, 200), (0, 0, width, height), border_radius=5)
        self._screen.blit(graph_surface, (x, y))
        
        # Draw threshold line
        threshold_y = int(height - (self._ear_threshold / 0.5) * height)
        pygame.draw.line(
            self._screen,
            (255, 100, 100, 150),
            (x, y + threshold_y),
            (x + width, y + threshold_y),
            1
        )
        
        # Draw EAR line
        if len(ear_history) >= 2:
            points = []
            for i, ear_val in enumerate(ear_history[-self._ear_graph_samples:]):
                px = x + int(i / self._ear_graph_samples * width)
                # Normalize EAR (typically 0 to 0.5)
                normalized = min(1.0, ear_val / 0.5)
                py = y + height - int(normalized * height)
                points.append((px, py))
            
            if len(points) >= 2:
                pygame.draw.lines(self._screen, (100, 255, 150), False, points, 2)
        
        # Label
        if self._small_font:
            label = self._small_font.render("EAR", True, (150, 150, 150))
            self._screen.blit(label, (x + 5, y + 2))
    
    def _draw_trail(self, trail_system: TrailSystem, velocity: float) -> None:
        """Draw the trail/afterimage effect."""
        if self._screen is None:
            return
        
        trail_points = trail_system.get_trail_points()
        dynamic_length = trail_system.get_dynamic_trail_length(velocity)
        
        # Only draw the last N points based on velocity
        points_to_draw = trail_points[-dynamic_length:] if len(trail_points) > dynamic_length else trail_points
        
        for x, y, radius, alpha, color in points_to_draw:
            # Create a surface with per-pixel alpha for smooth blending
            size = int(radius * 2 + 4)
            trail_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            
            # Draw circle with alpha
            pygame.draw.circle(
                trail_surface,
                color,  # Already includes alpha
                (size // 2, size // 2),
                int(radius)
            )
            
            # Blit to screen
            self._screen.blit(
                trail_surface,
                (int(x - size // 2), int(y - size // 2))
            )
    
    def _draw_gaze_circle(self, x: float, y: float) -> None:
        """Draw the main gaze indicator circle."""
        if self._screen is None:
            return
        
        # Draw filled circle
        pygame.draw.circle(
            self._screen,
            self._gaze_color,
            (int(x), int(y)),
            self._gaze_radius
        )
        
        # Draw outline for better visibility
        pygame.draw.circle(
            self._screen,
            (255, 255, 255),
            (int(x), int(y)),
            self._gaze_radius,
            2
        )
    
    def _draw_gaze_glow(self, x: float, y: float) -> None:
        """Draw a subtle glow effect around the gaze circle."""
        if self._screen is None:
            return
        
        # Create multiple concentric circles with decreasing alpha for glow
        glow_layers = [
            (self._gaze_radius + 15, 15),
            (self._gaze_radius + 10, 25),
            (self._gaze_radius + 5, 40),
        ]
        
        for radius, alpha in glow_layers:
            size = radius * 2 + 4
            glow_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surface,
                (*self._gaze_color, alpha),
                (size // 2, size // 2),
                radius
            )
            self._screen.blit(
                glow_surface,
                (int(x - size // 2), int(y - size // 2))
            )
    
    def _draw_info_panel(self, gaze_state: GazeState) -> None:
        """Draw the information panel in the top-right corner."""
        if self._screen is None or self._font is None:
            return
        
        # Panel position (top-right)
        panel_x = self._width - self._panel_width - self._panel_padding
        panel_y = self._panel_padding
        
        # Draw panel background with slight transparency
        panel_surface = pygame.Surface(
            (self._panel_width, self._panel_height),
            pygame.SRCALPHA
        )
        
        # Draw rounded rectangle background
        pygame.draw.rect(
            panel_surface,
            (*self._panel_bg_color, 220),
            (0, 0, self._panel_width, self._panel_height),
            border_radius=10
        )
        
        # Draw border
        pygame.draw.rect(
            panel_surface,
            (*self._panel_accent_color, 150),
            (0, 0, self._panel_width, self._panel_height),
            width=2,
            border_radius=10
        )
        
        self._screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw text content
        line_height = 28
        text_x = panel_x + 15
        text_y = panel_y + 15
        
        # Title
        title = self._font.render("GAZE TRACKER", True, self._panel_accent_color)
        self._screen.blit(title, (text_x, text_y))
        text_y += line_height + 5
        
        # Gaze position
        pos_text = f"Position: ({int(gaze_state.x)}, {int(gaze_state.y)})"
        pos_surface = self._font.render(pos_text, True, self._panel_text_color)
        self._screen.blit(pos_surface, (text_x, text_y))
        text_y += line_height
        
        # Velocity
        vel_text = f"Velocity: {int(gaze_state.velocity_magnitude)} px/s"
        vel_surface = self._font.render(vel_text, True, self._panel_text_color)
        self._screen.blit(vel_surface, (text_x, text_y))
        text_y += line_height
        
        # Acceleration
        accel_text = f"Accel: {int(gaze_state.acceleration_magnitude)} px/s²"
        accel_surface = self._font.render(accel_text, True, self._panel_text_color)
        self._screen.blit(accel_surface, (text_x, text_y))
        text_y += line_height
        
        # FPS
        fps_text = f"FPS: {int(self._current_fps)}"
        fps_surface = self._font.render(fps_text, True, self._panel_text_color)
        self._screen.blit(fps_surface, (text_x, text_y))
    
    def handle_events(self) -> bool:
        """
        Handle Pygame events.
        
        Returns:
            bool: True if should continue running, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_q:
                    return False
        
        return True
    
    def cleanup(self) -> None:
        """Clean up Pygame resources."""
        pygame.mouse.set_visible(True)
        pygame.quit()
        self._is_initialized = False
    
    @property
    def width(self) -> int:
        """Get screen width."""
        return self._width
    
    @property
    def height(self) -> int:
        """Get screen height."""
        return self._height
    
    @property
    def is_initialized(self) -> bool:
        """Check if renderer is initialized."""
        return self._is_initialized
