import numpy as np

"""
PID controller for the ball and beam system.
"""
class PIDController:
    def __init__(self, Kp=5.0, Ki=0.0, Kd=2.0, max_angle_deg=15):
        """Initialize the PID controller."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_angle = max_angle_deg
        
        self.prev_error = 0.0
        self.integral = 0.0
    
    def compute(self, error, velocity, dt):
        """Compute controller output based on error and time step."""
        # Proportional term
        P = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Derivative term (using provided velocity instead of error derivative)
        D = self.Kd * (-velocity)  # Negative velocity = positive error derivative
        
        # Total output
        output = P + I + D
        
        # Limit output to max angle
        output = np.clip(output, -self.max_angle, self.max_angle)
        
        # Update previous error
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset controller internal state."""
        self.prev_error = 0.0
        self.integral = 0.0
