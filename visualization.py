import pygame
import numpy as np
from fuzzy_viz import FuzzyMembershipViz

"""
Visualization system for the ball and beam control system.
"""
class BallAndBeamVisualization:
    
    def __init__(self, width=1200, height=800):
        """Initialize the visualization system."""
        self.width = width
        self.height = height
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ball and Beam Control System")
        
        # Simulation area (split screen for comparison)
        self.fuzzy_area = pygame.Rect(0, 0, width//2, height*2//3)
        self.pid_area = pygame.Rect(width//2, 0, width//2, height*2//3)
        
        # Metrics and visualization area
        self.metrics_area = pygame.Rect(0, height*2//3, width, height//3)
        
        # Colors
        self.colors = {
            'background': (240, 240, 240),
            'beam': (150, 150, 150),
            'ball': (255, 50, 50),
            'target': (50, 255, 50),
            'fuzzy': (100, 100, 255),
            'pid': (255, 100, 100),
            'text': (0, 0, 0)
        }
        
        # Font
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Performance metrics
        self.metrics = {
            'fuzzy': {
                'position': [],
                'error': [],
                'angle': [],
                'time': []
            },
            'pid': {
                'position': [],
                'error': [],
                'angle': [],
                'time': []
            }
        }
        
        # Time tracking
        self.current_time = 0.0
        
        # Membership function visualization
        self.mf_viz = FuzzyMembershipViz(self.metrics_area)

        # For tracking convergence time
        self.start_time = None
        self.converged = {'fuzzy': False, 'pid': False}
        self.convergence_times = {'fuzzy': None, 'pid': None}
        self.convergence_threshold = 0.01  # Consider converged when within 1cm of target
        self.velocity_threshold = 0.02     # And velocity is nearly zero
    
    def update(self, fuzzy_system, pid_system, target_position, time):
        """Update visualization based on current system states."""
        self.current_time = time
        
        # Update metrics
        self._update_metrics(fuzzy_system, pid_system, target_position)
        
        # Render everything
        self._render(fuzzy_system, pid_system, target_position)
    
    
    def _update_metrics(self, fuzzy_system, pid_system, target):
        """Update performance metrics."""
        # Fuzzy system metrics
        self.metrics['fuzzy']['position'].append(fuzzy_system.position)
        self.metrics['fuzzy']['error'].append(target - fuzzy_system.position)
        self.metrics['fuzzy']['angle'].append(fuzzy_system.angle)
        self.metrics['fuzzy']['time'].append(self.current_time)
        
        # PID system metrics
        self.metrics['pid']['position'].append(pid_system.position)
        self.metrics['pid']['error'].append(target - pid_system.position)
        self.metrics['pid']['angle'].append(pid_system.angle)
        self.metrics['pid']['time'].append(self.current_time)
    
    def _render(self, fuzzy_system, pid_system, target_position):
        """Render the entire visualization."""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw fuzzy system
        self._draw_system(self.fuzzy_area, fuzzy_system, target_position, "Fuzzy Logic Control")
        
        # Draw PID system
        self._draw_system(self.pid_area, pid_system, target_position, "PID Control")
        
        # Draw metrics
        self._draw_metrics()
        
        # Draw membership function visualization
        self.mf_viz.draw(self.screen)
        
        # Update display
        pygame.display.flip()
    
    def _draw_system(self, area, system, target_position, title):
        """Draw a ball and beam system in the specified area."""
        # Draw title
        title_text = self.font.render(title, True, self.colors['text'])
        self.screen.blit(title_text, (area.x + 10, area.y + 10))
        
        # Calculate area center - this is our fulcrum position
        fulcrum_x = area.x + area.width // 2
        fulcrum_y = area.y + area.height // 2
        
        # Beam parameters
        beam_length = area.width * 0.8
        beam_thickness = 10
        
        # Create beam surface
        beam_surface = pygame.Surface((beam_length, beam_thickness), pygame.SRCALPHA)
        beam_surface.fill(self.colors['beam'])
        
        # Rotate beam around its center
        angle_deg = np.degrees(system.angle)
        rotated_beam = pygame.transform.rotate(beam_surface, -angle_deg)
        
        # Position beam with the center at fulcrum
        beam_rect = rotated_beam.get_rect(center=(fulcrum_x, fulcrum_y))
        self.screen.blit(rotated_beam, beam_rect)
        
        # Draw fulcrum point
        pygame.draw.circle(self.screen, (0, 150, 0), (fulcrum_x, fulcrum_y), 4)
        
        # Calculate ball position on beam in local coordinates
        # Map from [-0.5, 0.5] position range to pixels
        relative_pos = (system.position / system.position_max) * (beam_length / 2)
        
        # Calculate ball position after rotation
        angle_rad = system.angle
        ball_x = fulcrum_x + relative_pos * np.cos(angle_rad)
        ball_y = fulcrum_y + relative_pos * np.sin(angle_rad)
        
        # Draw ball
        ball_radius = 15
        pygame.draw.circle(self.screen, self.colors['ball'], (int(ball_x), int(ball_y)), ball_radius)
        
        # Draw target position
        target_rel_pos = (target_position / system.position_max) * (beam_length / 2)
        target_x = fulcrum_x + target_rel_pos
        pygame.draw.circle(self.screen, self.colors['target'], (int(target_x), fulcrum_y), 5)
        
        # Display metrics for this system
        controller_type = 'fuzzy' if 'Fuzzy' in title else 'pid'
        
        metrics_text = [
            f"Position: {system.position:.3f} m",
            f"Velocity: {system.velocity:.3f} m/s",
            f"Angle: {np.degrees(system.angle):.2f}°",
            f"Error: {target_position - system.position:.3f} m",
            f"Time: {self.current_time:.2f} s"
        ]
        
        # Add convergence time if converged
        if hasattr(self, 'converged') and self.converged.get(controller_type):
            metrics_text.append(f"Converged in: {self.convergence_times[controller_type]:.2f} s")
        
        for i, text in enumerate(metrics_text):
            render_text = self.font.render(text, True, self.colors['text'])
            self.screen.blit(render_text, (area.x + 10, area.y + 40 + i * 20))
    
    def _draw_metrics(self):
        """Draw performance metrics and plots."""
        # Draw border
        pygame.draw.rect(self.screen, (200, 200, 200), self.metrics_area, 2)
        
        # TODO: Implement plots and performance metrics
        # This would include:
        # - Position vs time plot
        # - Error vs time plot
        # - Calculated metrics (overshoot, settling time, etc.)
        
        # For now, just display basic text
        title_text = self.font.render("Performance Metrics", True, self.colors['text'])
        self.screen.blit(title_text, (self.metrics_area.x + 10, self.metrics_area.y + 10))
