import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class BallAndBeamSystem:
    def __init__(self, m=0.11, R=0.015, g=9.8, J=9.99e-6, L=1.0, d=0.03):
        # Ball mass (kg)
        self.m = m
        # Ball radius (m)
        self.R = R
        # Gravity (m/s^2)
        self.g = g
        # Ball's moment of inertia
        self.J = J
        # Beam length (m)
        self.L = L
        # Distance from pivot to beam (m)
        self.d = d
        
    def dynamics(self, t, state, alpha):
        """Nonlinear dynamics of the ball and beam system"""
        r, r_dot = state
        
        # Nonlinear equation of motion for the ball
        inertia_term = self.J / (self.m * self.R**2) + 1
        r_ddot = (self.g * np.sin(alpha) - r * (alpha**2)) / inertia_term
        
        return [r_dot, r_ddot]
    
    def simulate(self, initial_state, control_func, t_span, dt=0.01):
        """Simulate the system with a given control function"""
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        def dynamics_with_control(t, state):
            r, r_dot = state
            alpha = control_func(t, r, r_dot)
            return self.dynamics(t, state, alpha)
        
        # Solve the ODE
        solution = solve_ivp(
            dynamics_with_control, 
            t_span, 
            initial_state, 
            t_eval=t_eval, 
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        return solution.t, solution.y

class CorrectFuzzyController:
    def __init__(self, setpoint=0.5):
        self.setpoint = setpoint
        self.debug_output = False  # Toggle for detailed debugging
        
        # Create antecedents and consequent with sufficient universe range
        error = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'error')
        d_error = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'delta_error')
        angle = ctrl.Consequent(np.linspace(-0.4, 0.4, 1000), 'angle')
        
        # Define membership functions for better control granularity
        # Using 5 linguistic variables for smoother control transitions
        error['NB'] = fuzz.trimf(error.universe, [-1.0, -1.0, -0.5])     # Negative Big
        error['NS'] = fuzz.trimf(error.universe, [-0.7, -0.3, 0.0])      # Negative Small
        error['ZE'] = fuzz.trimf(error.universe, [-0.2, 0.0, 0.2])       # Zero
        error['PS'] = fuzz.trimf(error.universe, [0.0, 0.3, 0.7])        # Positive Small
        error['PB'] = fuzz.trimf(error.universe, [0.5, 1.0, 1.0])        # Positive Big
        
        d_error['NB'] = fuzz.trimf(d_error.universe, [-1.0, -1.0, -0.5]) # Negative Big
        d_error['NS'] = fuzz.trimf(d_error.universe, [-0.7, -0.3, 0.0])  # Negative Small
        d_error['ZE'] = fuzz.trimf(d_error.universe, [-0.2, 0.0, 0.2])   # Zero
        d_error['PS'] = fuzz.trimf(d_error.universe, [0.0, 0.3, 0.7])    # Positive Small
        d_error['PB'] = fuzz.trimf(d_error.universe, [0.5, 1.0, 1.0])    # Positive Big
        
        angle['NB'] = fuzz.trimf(angle.universe, [-0.4, -0.4, -0.25])    # Negative Big
        angle['NS'] = fuzz.trimf(angle.universe, [-0.3, -0.15, 0.0])     # Negative Small
        angle['ZE'] = fuzz.trimf(angle.universe, [-0.1, 0.0, 0.1])       # Zero
        angle['PS'] = fuzz.trimf(angle.universe, [0.0, 0.15, 0.3])       # Positive Small
        angle['PB'] = fuzz.trimf(angle.universe, [0.25, 0.4, 0.4])       # Positive Big
        
        # Create rule base with CORRECT physics understanding
        # IMPORTANT: For ball and beam, POSITIVE angle tilts beam to make ball move POSITIVE
        # When error is positive (setpoint > position), need positive angle to move ball right
        rules = [
            # NB (Negative Big error) - Ball is far right of setpoint
            ctrl.Rule(error['NB'] & d_error['NB'], angle['NB']),  # Moving away fast → Strong negative tilt
            ctrl.Rule(error['NB'] & d_error['NS'], angle['NB']),  # Moving away slowly → Strong negative tilt
            ctrl.Rule(error['NB'] & d_error['ZE'], angle['NB']),  # Not moving → Strong negative tilt
            ctrl.Rule(error['NB'] & d_error['PS'], angle['NS']),  # Moving toward slowly → Medium negative tilt
            ctrl.Rule(error['NB'] & d_error['PB'], angle['ZE']),  # Moving toward fast → No tilt (let momentum work)
            
            # NS (Negative Small error) - Ball is slightly right of setpoint
            ctrl.Rule(error['NS'] & d_error['NB'], angle['NB']),  # Moving away fast → Strong negative tilt
            ctrl.Rule(error['NS'] & d_error['NS'], angle['NS']),  # Moving away slowly → Medium negative tilt
            ctrl.Rule(error['NS'] & d_error['ZE'], angle['NS']),  # Not moving → Medium negative tilt
            ctrl.Rule(error['NS'] & d_error['PS'], angle['ZE']),  # Moving toward slowly → No tilt
            ctrl.Rule(error['NS'] & d_error['PB'], angle['PS']),  # Moving toward fast → Slight dampening
            
            # ZE (Zero error) - Ball is at setpoint
            ctrl.Rule(error['ZE'] & d_error['NB'], angle['NS']),  # Moving right fast → Medium negative tilt
            ctrl.Rule(error['ZE'] & d_error['NS'], angle['NS']),  # Moving right slowly → Medium negative tilt
            ctrl.Rule(error['ZE'] & d_error['ZE'], angle['ZE']),  # Not moving → Keep level (perfect!)
            ctrl.Rule(error['ZE'] & d_error['PS'], angle['PS']),  # Moving left slowly → Medium positive tilt
            ctrl.Rule(error['ZE'] & d_error['PB'], angle['PS']),  # Moving left fast → Medium positive tilt
            
            # PS (Positive Small error) - Ball is slightly left of setpoint
            ctrl.Rule(error['PS'] & d_error['NB'], angle['NS']),  # Moving away fast → Slight dampening
            ctrl.Rule(error['PS'] & d_error['NS'], angle['ZE']),  # Moving away slowly → No tilt
            ctrl.Rule(error['PS'] & d_error['ZE'], angle['PS']),  # Not moving → Medium positive tilt
            ctrl.Rule(error['PS'] & d_error['PS'], angle['PS']),  # Moving toward slowly → Medium positive tilt
            ctrl.Rule(error['PS'] & d_error['PB'], angle['PB']),  # Moving toward fast → Strong positive tilt
            
            # PB (Positive Big error) - Ball is far left of setpoint
            ctrl.Rule(error['PB'] & d_error['NB'], angle['ZE']),  # Moving away fast → No tilt (let momentum work)
            ctrl.Rule(error['PB'] & d_error['NS'], angle['PS']),  # Moving away slowly → Medium positive tilt
            ctrl.Rule(error['PB'] & d_error['ZE'], angle['PB']),  # Not moving → Strong positive tilt
            ctrl.Rule(error['PB'] & d_error['PS'], angle['PB']),  # Moving toward slowly → Strong positive tilt
            ctrl.Rule(error['PB'] & d_error['PB'], angle['PB'])   # Moving toward fast → Strong positive tilt
        ]
        
        # Create control system with centroid defuzzification for smoother control
        self.control_system = ctrl.ControlSystem(rules)
        self.controller = ctrl.ControlSystemSimulation(self.control_system)
        
        self.counter = 0
        
    def compute_control(self, t, position, velocity):
        """Compute control action (beam angle) based on ball position and velocity"""
        # Calculate error (positive error means ball should move right)
        error = self.setpoint - position
        
        # Calculate change in error (negative velocity means error is decreasing)
        delta_error = -velocity
        
        # Scale inputs with appropriate scaling factors
        error_scaled = np.clip(error / 0.5, -1, 1)
        delta_error_scaled = np.clip(delta_error / 1.0, -1, 1)
        
        # Print diagnostic information periodically
        self.counter += 1
        if self.counter % 50 == 0 or self.debug_output:
            print(f"Time: {t:.2f}s | Pos: {position:.4f}m | Error: {error:.4f}m | Vel: {velocity:.4f}m/s")
            print(f"Scaled Error: {error_scaled:.4f} | Scaled dError: {delta_error_scaled:.4f}")
        
        # Compute fuzzy control output
        try:
            self.controller.input['error'] = error_scaled
            self.controller.input['delta_error'] = delta_error_scaled
            self.controller.compute()
            angle = self.controller.output['angle']
            
            if self.counter % 50 == 0 or self.debug_output:
                print(f"Control output: Beam angle = {angle:.4f} rad\n")
                
            return angle
        except Exception as e:
            print(f"Error in fuzzy computation: {e}")
            # Fallback control - proportional to error
            return np.clip(error * 0.4, -0.4, 0.4)

class OptimizedPIDController:
    def __init__(self, kp=0.7, ki=0.05, kd=1.2, setpoint=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.error_sum = 0
        self.last_error = 0
        self.last_time = 0
        self.first_call = True
        self.counter = 0
        
    def compute_control(self, t, position, velocity):
        """Compute the control action (beam angle) based on ball position and velocity"""
        # Calculate error
        error = self.setpoint - position
        
        # Calculate time step
        if self.first_call:
            dt = 0.01
            self.first_call = False
        else:
            dt = t - self.last_time
            dt = max(dt, 0.001)  # Prevent division by zero
        
        # P term
        p_term = self.kp * error
        
        # I term with anti-windup
        self.error_sum += error * dt
        self.error_sum = np.clip(self.error_sum, -1.0, 1.0)  # Prevent integral windup
        i_term = self.ki * self.error_sum
        
        # D term (using provided velocity for better performance)
        d_term = self.kd * (-velocity)
        
        # Compute control output
        alpha = p_term + i_term + d_term
        
        # Limit control output to reasonable range
        alpha = np.clip(alpha, -0.4, 0.4)
        
        # Print diagnostic information periodically
        self.counter += 1
        if self.counter % 50 == 0:
            print(f"Time: {t:.2f}s | Pos: {position:.4f}m | Error: {error:.4f}m | Vel: {velocity:.4f}m/s")
            print(f"P term: {p_term:.4f} | I term: {i_term:.4f} | D term: {d_term:.4f}")
            print(f"PID output: Beam angle = {alpha:.4f} rad\n")
        
        # Update last values
        self.last_error = error
        self.last_time = t
        
        return alpha

def test_and_analyze_controller(system, controller, setpoint=0.5, simulation_time=5.0):
    """Test a controller and analyze its performance"""
    # Define initial state and control function
    initial_state = [0.0, 0.0]  # Start at zero position, zero velocity
    
    def control_func(t, r, r_dot):
        return controller.compute_control(t, r, r_dot)
    
    # Simulate the system
    print(f"\nRunning simulation with {controller.__class__.__name__}...")
    t, y = system.simulate(initial_state, control_func, (0, simulation_time))
    
    # Extract position and velocity data
    positions = y[0]
    velocities = y[1]
    
    # Calculate performance metrics
    settling_time = calculate_settling_time(t, positions, setpoint)
    overshoot = calculate_overshoot(positions, setpoint)
    steady_state_error = abs(positions[-1] - setpoint)
    
    # Print performance metrics
    print(f"\nPerformance Metrics:")
    print(f"Settling Time: {settling_time:.3f} s")
    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Steady-State Error: {steady_state_error:.6f} m")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, positions, 'b-', label='Ball Position')
    plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    plt.axhline(y=setpoint*1.02, color='g', linestyle=':', label='+2% Band')
    plt.axhline(y=setpoint*0.98, color='g', linestyle=':', label='-2% Band')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title(f'{controller.__class__.__name__}: Ball Position vs Time')
    
    plt.subplot(3, 1, 2)
    plt.plot(t, velocities, 'g-', label='Ball Velocity')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Ball Velocity vs Time')
    
    # Add control input plot
    control_inputs = np.array([control_func(t_i, pos, vel) for t_i, pos, vel in zip(t, positions, velocities)])
    plt.subplot(3, 1, 3)
    plt.plot(t, control_inputs, 'r-', label='Beam Angle')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Control Input (Beam Angle) vs Time')
    
    plt.tight_layout()
    plt.show()
    
    return t, positions, velocities, control_inputs, (settling_time, overshoot, steady_state_error)

def calculate_settling_time(t, positions, setpoint, threshold=0.02):
    """Calculate the settling time (time to reach and stay within ±2% of setpoint)"""
    threshold_band = threshold * abs(setpoint)
    
    if threshold_band == 0:
        threshold_band = 0.01  # Use absolute threshold if setpoint is zero
    
    for i in range(len(t)-20):  # Check if position stays in band for at least 20 samples
        if abs(positions[i] - setpoint) <= threshold_band:
            # Check if position stays within band for next 20 samples
            if all(abs(positions[j] - setpoint) <= threshold_band for j in range(i, min(i+20, len(positions)))):
                return t[i]
    
    # If never settles, return the full simulation time as a penalty
    return t[-1]

def calculate_overshoot(positions, setpoint):
    """Calculate the percentage overshoot"""
    if abs(setpoint) < 1e-6:
        # Special case for zero setpoint
        max_abs_position = max(abs(p) for p in positions)
        return max_abs_position * 100 if max_abs_position > 0 else 0
    
    # Find index where position first crosses setpoint
    first_crossing_idx = None
    for i in range(len(positions)-1):
        if (positions[i] < setpoint and positions[i+1] >= setpoint) or \
           (positions[i] > setpoint and positions[i+1] <= setpoint):
            first_crossing_idx = i+1
            break
    
    if first_crossing_idx is None:
        # If position never crosses setpoint
        return 0
    
    # Find maximum deviation from setpoint after first crossing
    if setpoint > 0:
        max_position = max(positions[first_crossing_idx:])
        if max_position > setpoint:
            return (max_position - setpoint) / setpoint * 100
    else:
        min_position = min(positions[first_crossing_idx:])
        if min_position < setpoint:
            return (setpoint - min_position) / abs(setpoint) * 100
    
    return 0

def run_tests():
    # Create the ball and beam system
    system = BallAndBeamSystem()
    
    # Test setpoint
    setpoint = 0.5  # meters
    
    # Create and test the corrected fuzzy controller
    print("\n--- Testing Corrected Fuzzy Controller ---")
    fuzzy_controller = CorrectFuzzyController(setpoint=setpoint)
    fuzzy_controller.debug_output = True  # Enable detailed debugging for first few steps
    fuzzy_results = test_and_analyze_controller(system, fuzzy_controller, setpoint)
    fuzzy_controller.debug_output = False  # Disable for clarity afterwards
    
    # Create and test an optimized PID controller for comparison
    print("\n--- Testing Optimized PID Controller (Benchmark) ---")
    pid_controller = OptimizedPIDController(setpoint=setpoint)
    pid_results = test_and_analyze_controller(system, pid_controller, setpoint)
    
    return fuzzy_results, pid_results

if __name__ == "__main__":
    run_tests()


    