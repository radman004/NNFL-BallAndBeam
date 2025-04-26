import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""
Fuzzy logic controller for the ball and beam system.
"""
class FuzzyController:
    def __init__(self, beam_length=1.0, max_angle_deg=15, max_velocity=1.0):
        """Initialize the fuzzy logic controller."""
        # Set up range variables
        self.beam_length = beam_length
        self.max_angle = max_angle_deg
        self.max_velocity = max_velocity
        
        # Create the fuzzy control system
        self._create_fuzzy_system()
    
    def _create_fuzzy_system(self):
        """Create the fuzzy inference system with rules."""
        # Universe variables
        # Position error universe - use a smaller range for more precision
        self.error = ctrl.Antecedent(np.linspace(-self.beam_length/2, self.beam_length/2, 100), 'error')
        
        # Velocity universe
        self.velocity = ctrl.Antecedent(np.linspace(-self.max_velocity, self.max_velocity, 100), 'velocity')
        
        # Angle universe (output)
        self.angle = ctrl.Consequent(np.linspace(-self.max_angle, self.max_angle, 100), 'angle')
        
        # OPTIMIZED MEMBERSHIP FUNCTIONS FOR ERROR
        # More sensitive near zero, wider at extremes
        self.error['negative_large'] = fuzz.trapmf(self.error.universe, 
                                                 [-self.beam_length/2, -self.beam_length/2, -self.beam_length/4, -self.beam_length/8])
        self.error['negative_small'] = fuzz.trimf(self.error.universe, 
                                                [-self.beam_length/6, -self.beam_length/12, 0])
        self.error['zero'] = fuzz.trimf(self.error.universe, 
                                      [-self.beam_length/20, 0, self.beam_length/20])
        self.error['positive_small'] = fuzz.trimf(self.error.universe, 
                                                [0, self.beam_length/12, self.beam_length/6])
        self.error['positive_large'] = fuzz.trapmf(self.error.universe, 
                                                 [self.beam_length/8, self.beam_length/4, self.beam_length/2, self.beam_length/2])
        
        # OPTIMIZED MEMBERSHIP FUNCTIONS FOR VELOCITY
        # More granular velocity control
        self.velocity['negative_large'] = fuzz.trapmf(self.velocity.universe, 
                                                    [-self.max_velocity, -self.max_velocity, -self.max_velocity/2, -self.max_velocity/4])
        self.velocity['negative_small'] = fuzz.trimf(self.velocity.universe, 
                                                   [-self.max_velocity/3, -self.max_velocity/6, 0])
        self.velocity['zero'] = fuzz.trimf(self.velocity.universe, 
                                         [-self.max_velocity/10, 0, self.max_velocity/10])
        self.velocity['positive_small'] = fuzz.trimf(self.velocity.universe, 
                                                   [0, self.max_velocity/6, self.max_velocity/3])
        self.velocity['positive_large'] = fuzz.trapmf(self.velocity.universe, 
                                                    [self.max_velocity/4, self.max_velocity/2, self.max_velocity, self.max_velocity])
        
        # OPTIMIZED MEMBERSHIP FUNCTIONS FOR ANGLE OUTPUT
        # More precise near zero for fine adjustments
        self.angle['negative_large'] = fuzz.trapmf(self.angle.universe, 
                                                 [-self.max_angle, -self.max_angle, -self.max_angle/2, -self.max_angle/4])
        self.angle['negative_medium'] = fuzz.trimf(self.angle.universe, 
                                                 [-self.max_angle/3, -self.max_angle/5, -self.max_angle/10])
        self.angle['negative_small'] = fuzz.trimf(self.angle.universe, 
                                                [-self.max_angle/8, -self.max_angle/16, 0])
        self.angle['zero'] = fuzz.trimf(self.angle.universe, 
                                      [-self.max_angle/20, 0, self.max_angle/20])
        self.angle['positive_small'] = fuzz.trimf(self.angle.universe, 
                                                [0, self.max_angle/16, self.max_angle/8])
        self.angle['positive_medium'] = fuzz.trimf(self.angle.universe, 
                                                 [self.max_angle/10, self.max_angle/5, self.max_angle/3])
        self.angle['positive_large'] = fuzz.trapmf(self.angle.universe, 
                                                 [self.max_angle/4, self.max_angle/2, self.max_angle, self.max_angle])
        
        # IMPROVED RULE BASE WITH DAMPING BEHAVIOR
        # Basic position control rules
        rule1 = ctrl.Rule(self.error['negative_large'], self.angle['negative_large'])
        rule2 = ctrl.Rule(self.error['negative_small'], self.angle['negative_small'])
        rule3 = ctrl.Rule(self.error['zero'] & self.velocity['zero'], self.angle['zero'])
        rule4 = ctrl.Rule(self.error['positive_small'], self.angle['positive_small'])
        rule5 = ctrl.Rule(self.error['positive_large'], self.angle['positive_large'])
        
        # Damping rules to reduce oscillation
        rule6 = ctrl.Rule(self.error['negative_small'] & self.velocity['negative_large'], self.angle['negative_medium'])
        rule7 = ctrl.Rule(self.error['negative_small'] & self.velocity['negative_small'], self.angle['zero'])
        rule8 = ctrl.Rule(self.error['zero'] & self.velocity['negative_small'], self.angle['positive_small'])
        rule9 = ctrl.Rule(self.error['zero'] & self.velocity['negative_large'], self.angle['positive_medium'])
        
        rule10 = ctrl.Rule(self.error['positive_small'] & self.velocity['positive_large'], self.angle['positive_medium'])
        rule11 = ctrl.Rule(self.error['positive_small'] & self.velocity['positive_small'], self.angle['zero'])
        rule12 = ctrl.Rule(self.error['zero'] & self.velocity['positive_small'], self.angle['negative_small'])
        rule13 = ctrl.Rule(self.error['zero'] & self.velocity['positive_large'], self.angle['negative_medium'])
        
        # Create control system
        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, 
                                                 rule6, rule7, rule8, rule9, 
                                                 rule10, rule11, rule12, rule13])
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def compute(self, error, velocity):
        """Compute controller output based on position error and velocity."""
        # Clip inputs to universe ranges
        error_clipped = np.clip(error, -self.beam_length/2, self.beam_length/2)
        velocity_clipped = np.clip(velocity, -self.max_velocity, self.max_velocity)
        
        # Set inputs
        self.simulation.input['error'] = error_clipped
        self.simulation.input['velocity'] = velocity_clipped
        
        try:
            # Compute
            self.simulation.compute()
            
            # Get output
            angle = self.simulation.output['angle']
            
            # Apply a scaling factor for finer control near target
            # This makes the controller more aggressive for large errors
            # and more precise for small errors
            if abs(error) < 0.05:
                # Fine control near target
                scale_factor = 0.8
            else:
                # More aggressive for larger errors
                scale_factor = 1.2
                
            return angle * scale_factor
            
        except:
            # Fallback in case of computational errors
            print("Fuzzy computation error, using fallback")
            # Simple P controller as fallback
            return np.clip(error * 10.0, -self.max_angle, self.max_angle)
