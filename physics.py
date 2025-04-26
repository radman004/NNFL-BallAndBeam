import numpy as np

"""
Physics model for the ball and beam system.
Implements realistic dynamics with proper constraints.
"""
class BallAndBeamSystem:
    def __init__(self, beam_length=1.0, max_angle_deg=15, ball_mass=0.1, friction=0.1):
        """Initialize the ball and beam system with physical parameters."""
        # Physical parameters
        self.beam_length = beam_length  # Total length of beam (m)
        self.max_angle = np.radians(max_angle_deg)  # Maximum beam angle (rad)
        self.ball_mass = ball_mass  # Mass of the ball (kg)
        self.ball_radius = 0.05  # Radius of the ball (m)
        self.friction = friction  # Friction coefficient
        self.g = 9.81  # Gravity (m/s²)
        
        # State variables
        self.position = 0.0  # Ball position along beam (m)
        self.velocity = 0.0  # Ball velocity (m/s)
        self.angle = 0.0  # Current beam angle (rad)
        
        # Constraints
        self.position_min = -beam_length/2  # Minimum position (m)
        self.position_max = beam_length/2   # Maximum position (m)
    
    def update(self, angle, dt):
        """Update system state based on beam angle and time step."""
        # Limit angle to max allowed
        self.angle = np.clip(angle, -self.max_angle, self.max_angle)
        
        # Calculate acceleration
        # a = g*sin(angle) - friction*velocity
        acceleration = self.g * np.sin(self.angle) - self.friction * self.velocity
        
        # Update velocity
        self.velocity += acceleration * dt
        
        # Update position
        self.position += self.velocity * dt
        
        # Apply position constraints (ball stays on beam)
        self.position = np.clip(self.position, self.position_min, self.position_max)
        
        # If ball hits edge of beam, stop it
        if self.position == self.position_min or self.position == self.position_max:
            self.velocity = 0.0
        
        return self.position, self.velocity
    
    def reset(self, position=0.0, velocity=0.0):
        """Reset system to initial state."""
        self.position = position
        self.velocity = velocity
        self.angle = 0.0

