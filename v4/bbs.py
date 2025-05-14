import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
from datetime import datetime
import pandas as pd
import logging
import sys

class BallAndBeamSystem:
    def __init__(self, m=0.11, R=0.015, g=9.8, J=9.99e-6, L=1.0, d=0.03):
        """
        Initialize the ball and beam physical system.
        
        Parameters:
        -----------
        m : float
            Ball mass (kg)
        R : float
            Ball radius (m)
        g : float
            Gravity acceleration (m/s^2)
        J : float
            Ball's moment of inertia (kg*m^2)
        L : float
            Beam length (m)
        d : float
            Distance from pivot to beam (m)
        """
        self.m = m
        self.R = R
        self.g = g
        self.J = J
        self.L = L
        self.d = d
        
    def dynamics(self, t, state, alpha):
        """
        Implements the nonlinear dynamics of the ball and beam system.
        
        Parameters:
        -----------
        t : float
            Current time
        state : list
            Current state [position, velocity]
        alpha : float
            Beam angle in radians
        
        Returns:
        --------
        list
            State derivatives [velocity, acceleration]
        """
        r, r_dot = state
        
        # Nonlinear equation of motion for the ball
        inertia_term = self.J / (self.m * self.R**2) + 1
        r_ddot = (self.g * np.sin(alpha) - r * (alpha**2)) / inertia_term
        
        return [r_dot, r_ddot]
    
    def simulate(self, initial_state, control_func, t_span, dt=0.01):
        """
        Simulate the system with a given control function.
        
        Parameters:
        -----------
        initial_state : list
            Initial state [position, velocity]
        control_func : function
            Control function taking (t, position, velocity) and returning angle
        t_span : tuple
            Time span for simulation (t_start, t_end)
        dt : float
            Time step for simulation output
            
        Returns:
        --------
        tuple
            Time array and state history arrays
        """
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

class EnhancedFuzzyController:
    def __init__(self, setpoint=0.5):
        """
        Enhanced fuzzy logic controller for the ball and beam system.
        
        Parameters:
        -----------
        setpoint : float
            Desired ball position
        """
        self.setpoint = setpoint
        self.debug_output = False
        self.counter = 0
        
        # Create antecedents and consequent
        error = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'error')
        d_error = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'delta_error')
        angle = ctrl.Consequent(np.linspace(-0.4, 0.4, 1000), 'angle')
        
        # Enhanced membership functions with better overlap and finer control
        # Error membership functions
        error['NB'] = fuzz.trimf(error.universe, [-1.0, -1.0, -0.5])     # Negative Big
        error['NM'] = fuzz.trimf(error.universe, [-0.75, -0.5, -0.25])   # Negative Medium
        error['NS'] = fuzz.trimf(error.universe, [-0.4, -0.2, -0.05])    # Negative Small
        error['ZE'] = fuzz.trimf(error.universe, [-0.15, 0, 0.15])       # Zero (narrower)
        error['PS'] = fuzz.trimf(error.universe, [0.05, 0.2, 0.4])       # Positive Small
        error['PM'] = fuzz.trimf(error.universe, [0.25, 0.5, 0.75])      # Positive Medium
        error['PB'] = fuzz.trimf(error.universe, [0.5, 1.0, 1.0])        # Positive Big
        
        # Delta error membership functions
        d_error['NB'] = fuzz.trimf(d_error.universe, [-1.0, -1.0, -0.5]) # Negative Big
        d_error['NM'] = fuzz.trimf(d_error.universe, [-0.75, -0.5, -0.25]) # Negative Medium
        d_error['NS'] = fuzz.trimf(d_error.universe, [-0.4, -0.2, -0.05]) # Negative Small
        d_error['ZE'] = fuzz.trimf(d_error.universe, [-0.15, 0, 0.15])   # Zero
        d_error['PS'] = fuzz.trimf(d_error.universe, [0.05, 0.2, 0.4])   # Positive Small
        d_error['PM'] = fuzz.trimf(d_error.universe, [0.25, 0.5, 0.75])  # Positive Medium
        d_error['PB'] = fuzz.trimf(d_error.universe, [0.5, 1.0, 1.0])    # Positive Big
        
        # Angle membership functions
        angle['NB'] = fuzz.trimf(angle.universe, [-0.4, -0.4, -0.25])    # Negative Big
        angle['NM'] = fuzz.trimf(angle.universe, [-0.3, -0.2, -0.1])     # Negative Medium
        angle['NS'] = fuzz.trimf(angle.universe, [-0.15, -0.075, 0])     # Negative Small
        angle['ZE'] = fuzz.trimf(angle.universe, [-0.05, 0, 0.05])       # Zero
        angle['PS'] = fuzz.trimf(angle.universe, [0, 0.075, 0.15])       # Positive Small
        angle['PM'] = fuzz.trimf(angle.universe, [0.1, 0.2, 0.3])        # Positive Medium
        angle['PB'] = fuzz.trimf(angle.universe, [0.25, 0.4, 0.4])       # Positive Big
        
        # Enhanced rule base with improved braking behavior
        rules = [
            # NB (Ball far to right of setpoint)
            ctrl.Rule(error['NB'] & d_error['NB'], angle['NB']),  # Moving away fast → Strong negative
            ctrl.Rule(error['NB'] & d_error['NM'], angle['NB']),  # Moving away medium → Strong negative
            ctrl.Rule(error['NB'] & d_error['NS'], angle['NB']),  # Moving away slow → Strong negative
            ctrl.Rule(error['NB'] & d_error['ZE'], angle['NB']),  # Not moving → Strong negative
            ctrl.Rule(error['NB'] & d_error['PS'], angle['NM']),  # Moving toward slow → Medium negative
            ctrl.Rule(error['NB'] & d_error['PM'], angle['NS']),  # Moving toward medium → Small negative
            ctrl.Rule(error['NB'] & d_error['PB'], angle['ZE']),  # Moving toward fast → Zero angle
            
            # NM (Ball moderately right of setpoint)
            ctrl.Rule(error['NM'] & d_error['NB'], angle['NB']),  # Moving away fast → Strong negative
            ctrl.Rule(error['NM'] & d_error['NM'], angle['NB']),  # Moving away medium → Strong negative
            ctrl.Rule(error['NM'] & d_error['NS'], angle['NM']),  # Moving away slow → Medium negative
            ctrl.Rule(error['NM'] & d_error['ZE'], angle['NM']),  # Not moving → Medium negative
            ctrl.Rule(error['NM'] & d_error['PS'], angle['NS']),  # Moving toward slow → Small negative
            ctrl.Rule(error['NM'] & d_error['PM'], angle['ZE']),  # Moving toward medium → Zero
            ctrl.Rule(error['NM'] & d_error['PB'], angle['PS']),  # Moving toward fast → Small positive (brake)
            
            # NS (Ball slightly right of setpoint)
            ctrl.Rule(error['NS'] & d_error['NB'], angle['NM']),  # Moving away fast → Medium negative
            ctrl.Rule(error['NS'] & d_error['NM'], angle['NM']),  # Moving away medium → Medium negative
            ctrl.Rule(error['NS'] & d_error['NS'], angle['NS']),  # Moving away slow → Small negative
            ctrl.Rule(error['NS'] & d_error['ZE'], angle['NS']),  # Not moving → Small negative
            ctrl.Rule(error['NS'] & d_error['PS'], angle['ZE']),  # Moving toward slow → Zero
            ctrl.Rule(error['NS'] & d_error['PM'], angle['PS']),  # Moving toward medium → Small positive (brake)
            ctrl.Rule(error['NS'] & d_error['PB'], angle['PM']),  # Moving toward fast → Medium positive (brake)
            
            # ZE (Ball at setpoint)
            ctrl.Rule(error['ZE'] & d_error['NB'], angle['NM']),  # Moving right fast → Medium negative (brake)
            ctrl.Rule(error['ZE'] & d_error['NM'], angle['NS']),  # Moving right medium → Small negative
            ctrl.Rule(error['ZE'] & d_error['NS'], angle['NS']),  # Moving right slow → Small negative
            ctrl.Rule(error['ZE'] & d_error['ZE'], angle['ZE']),  # Not moving → Zero (perfect!)
            ctrl.Rule(error['ZE'] & d_error['PS'], angle['PS']),  # Moving left slow → Small positive
            ctrl.Rule(error['ZE'] & d_error['PM'], angle['PS']),  # Moving left medium → Small positive
            ctrl.Rule(error['ZE'] & d_error['PB'], angle['PM']),  # Moving left fast → Medium positive (brake)
            
            # PS (Ball slightly left of setpoint)
            ctrl.Rule(error['PS'] & d_error['NB'], angle['NM']),  # Moving right fast → Medium negative (brake)
            ctrl.Rule(error['PS'] & d_error['NM'], angle['NS']),  # Moving right medium → Small negative (brake)
            ctrl.Rule(error['PS'] & d_error['NS'], angle['ZE']),  # Moving right slow → Zero
            ctrl.Rule(error['PS'] & d_error['ZE'], angle['PS']),  # Not moving → Small positive
            ctrl.Rule(error['PS'] & d_error['PS'], angle['PS']),  # Moving left slow → Small positive
            ctrl.Rule(error['PS'] & d_error['PM'], angle['PM']),  # Moving left medium → Medium positive
            ctrl.Rule(error['PS'] & d_error['PB'], angle['PM']),  # Moving left fast → Medium positive
            
            # PM (Ball moderately left of setpoint)
            ctrl.Rule(error['PM'] & d_error['NB'], angle['NS']),  # Moving right fast → Small negative (brake)
            ctrl.Rule(error['PM'] & d_error['NM'], angle['ZE']),  # Moving right medium → Zero
            ctrl.Rule(error['PM'] & d_error['NS'], angle['PS']),  # Moving right slow → Small positive
            ctrl.Rule(error['PM'] & d_error['ZE'], angle['PM']),  # Not moving → Medium positive
            ctrl.Rule(error['PM'] & d_error['PS'], angle['PM']),  # Moving left slow → Medium positive
            ctrl.Rule(error['PM'] & d_error['PM'], angle['PB']),  # Moving left medium → Strong positive
            ctrl.Rule(error['PM'] & d_error['PB'], angle['PB']),  # Moving left fast → Strong positive
            
            # PB (Ball far to left of setpoint)
            ctrl.Rule(error['PB'] & d_error['NB'], angle['ZE']),  # Moving right fast → Zero
            ctrl.Rule(error['PB'] & d_error['NM'], angle['PS']),  # Moving right medium → Small positive
            ctrl.Rule(error['PB'] & d_error['NS'], angle['PM']),  # Moving right slow → Medium positive
            ctrl.Rule(error['PB'] & d_error['ZE'], angle['PB']),  # Not moving → Strong positive
            ctrl.Rule(error['PB'] & d_error['PS'], angle['PB']),  # Moving left slow → Strong positive
            ctrl.Rule(error['PB'] & d_error['PM'], angle['PB']),  # Moving left medium → Strong positive
            ctrl.Rule(error['PB'] & d_error['PB'], angle['PB'])   # Moving left fast → Strong positive
        ]
        
        # Create control system with centroid defuzzification
        self.control_system = ctrl.ControlSystem(rules)
        self.controller = ctrl.ControlSystemSimulation(self.control_system)
        
    def compute_control(self, t, position, velocity):
        """
        Compute control action (beam angle) based on ball position and velocity.
        
        Parameters:
        -----------
        t : float
            Current time
        position : float
            Current ball position
        velocity : float
            Current ball velocity
            
        Returns:
        --------
        float
            Beam angle in radians
        """
        # Calculate error (positive error means ball should move right)
        error = self.setpoint - position
        
        # Calculate change in error (negative velocity means error is decreasing)
        delta_error = -velocity
        
        # Scale inputs with appropriate scaling factors
        # Use smaller error scale for better sensitivity near setpoint
        error_scaled = np.clip(error / 0.5, -1, 1)
        delta_error_scaled = np.clip(delta_error / 0.8, -1, 1)
        
        # Print diagnostic information periodically
        self.counter += 1
        if self.counter % 50 == 0 or self.debug_output:
            logging.info(f"Time: {t:.2f}s | Pos: {position:.4f}m | Error: {error:.4f}m | Vel: {velocity:.4f}m/s")
            logging.info(f"Scaled Error: {error_scaled:.4f} | Scaled dError: {delta_error_scaled:.4f}")
        
        # Compute fuzzy control output
        try:
            self.controller.input['error'] = error_scaled
            self.controller.input['delta_error'] = delta_error_scaled
            self.controller.compute()
            angle = self.controller.output['angle']
            
            if self.counter % 50 == 0 or self.debug_output:
                logging.info(f"Control output: Beam angle = {angle:.4f} rad\n")
                
            return angle
        except Exception as e:
            logging.info(f"Error in fuzzy computation: {e}")
            # Fallback control - proportional to error
            return np.clip(error * 0.3, -0.3, 0.3)


class ImprovedFuzzyController(EnhancedFuzzyController):
    def __init__(self, setpoint=0.5):
        """
        Enhanced fuzzy logic controller with minor improvements for stability.
        Inherits from the previous working version.
        """
        # Call the parent constructor to set up the basic controller
        super().__init__(setpoint)
        
        # We'll replace just the compute_control method with an improved version
        # The membership functions and rule base remain the same
        
    def compute_control(self, t, position, velocity):
        """
        Enhanced compute_control with better error handling and conservative limits
        """
        # Calculate error (positive error means ball should move right)
        error = self.setpoint - position
        
        # Calculate change in error (negative velocity means error is decreasing)
        delta_error = -velocity
        
        # Scale inputs with appropriate scaling factors
        error_scaled = np.clip(error / 0.5, -1, 1)
        delta_error_scaled = np.clip(delta_error / 0.8, -1, 1)
        
        # Print diagnostic information periodically
        self.counter += 1
        if self.counter % 50 == 0 or self.debug_output:
            logging.info(f"Time: {t:.2f}s | Pos: {position:.4f}m | Error: {error:.4f}m | Vel: {velocity:.4f}m/s")
            logging.info(f"Scaled Error: {error_scaled:.4f} | Scaled dError: {delta_error_scaled:.4f}")
        
        # Compute fuzzy control output with better error handling
        try:
            self.controller.input['error'] = error_scaled
            self.controller.input['delta_error'] = delta_error_scaled
            self.controller.compute()
            angle = self.controller.output['angle']
            
            # Apply more conservative limits to prevent extreme angles
            angle = np.clip(angle, -0.25, 0.25)
            
            if self.counter % 50 == 0 or self.debug_output:
                logging.info(f"Control output: Beam angle = {angle:.4f} rad\n")
                
            return angle
        except Exception as e:
            # More sophisticated fallback control
            # Use a basic PID-like fallback that's more robust
            p_term = 0.2 * error
            d_term = -0.3 * velocity
            fallback_angle = p_term + d_term
            
            # Apply conservative limits
            fallback_angle = np.clip(fallback_angle, -0.25, 0.25)
            
            logging.error(f"Error in fuzzy computation: {e}")
            logging.info(f"Using fallback control: Beam angle = {fallback_angle:.4f} rad\n")
                
            return fallback_angle


class OptimizedPIDController:
    def __init__(self, kp=0.7, ki=0.05, kd=1.2, setpoint=0.5):
        """
        PID controller for the ball and beam system.
        
        Parameters:
        -----------
        kp : float
            Proportional gain
        ki : float
            Integral gain
        kd : float
            Derivative gain
        setpoint : float
            Desired ball position
        """
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
        """
        Compute the control action (beam angle) based on ball position and velocity.
        
        Parameters:
        -----------
        t : float
            Current time
        position : float
            Current ball position
        velocity : float
            Current ball velocity
            
        Returns:
        --------
        float
            Beam angle in radians
        """
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
            logging.info(f"Time: {t:.2f}s | Pos: {position:.4f}m | Error: {error:.4f}m | Vel: {velocity:.4f}m/s")
            logging.info(f"P term: {p_term:.4f} | I term: {i_term:.4f} | D term: {d_term:.4f}")
            logging.info(f"PID output: Beam angle = {alpha:.4f} rad\n")
        
        # Update last values
        self.last_error = error
        self.last_time = t
        
        return alpha

def test_controller(system, controller, initial_state, setpoint, simulation_time=5.0, title=None):
    """
    Test a controller and analyze its performance.
    
    Parameters:
    -----------
    system : BallAndBeamSystem
        System to simulate
    controller : Controller object
        Controller to test
    initial_state : list
        Initial state [position, velocity]
    setpoint : float
        Desired ball position
    simulation_time : float
        Duration of simulation
    title : str, optional
        Title for the plot
        
    Returns:
    --------
    dict
        Test results including time, position, velocity, control inputs, and metrics
    """
    controller.setpoint = setpoint  # Set controller setpoint
    
    # Define control function to pass to the system
    def control_func(t, r, r_dot):
        return controller.compute_control(t, r, r_dot)
    
    # Simulate the system
    logging.info(f"\nRunning simulation with {controller.__class__.__name__}...")
    t, y = system.simulate(initial_state, control_func, (0, simulation_time))
    
    # Extract position and velocity data
    positions = y[0]
    velocities = y[1]
    
    # Calculate control inputs
    control_inputs = np.array([control_func(t_i, pos, vel) for t_i, pos, vel in zip(t, positions, velocities)])
    
    # Calculate performance metrics
    settling_time = calculate_settling_time(t, positions, setpoint)
    overshoot = calculate_overshoot(positions, setpoint)
    steady_state_error = abs(positions[-1] - setpoint)
    
    # Print performance metrics
    logging.info(f"\nPerformance Metrics:")
    logging.info(f"Settling Time: {settling_time:.3f} s")
    logging.info(f"Overshoot: {overshoot:.2f}%")
    logging.info(f"Steady-State Error: {steady_state_error:.6f} m")
    
    # Plot results
    if title:
        plot_title = f"{controller.__class__.__name__}: {title}"
    else:
        plot_title = f"{controller.__class__.__name__}"
        
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
    plt.title(f'{plot_title}: Ball Position vs Time')
    
    plt.subplot(3, 1, 2)
    plt.plot(t, velocities, 'g-', label='Ball Velocity')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Ball Velocity vs Time')
    
    plt.subplot(3, 1, 3)
    plt.plot(t, control_inputs, 'r-', label='Beam Angle')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Control Input (Beam Angle) vs Time')
    
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        "time": t,
        "position": positions,
        "velocity": velocities,
        "control": control_inputs,
        "metrics": {
            "settling_time": settling_time,
            "overshoot": overshoot,
            "steady_state_error": steady_state_error
        }
    }

def calculate_settling_time(t, positions, setpoint, threshold=0.02):
    """
    Calculate the settling time (time to reach and stay within ±2% of setpoint).
    
    Parameters:
    -----------
    t : array
        Time array
    positions : array
        Ball position array
    setpoint : float
        Desired ball position
    threshold : float
        Threshold percentage for settling
        
    Returns:
    --------
    float
        Settling time
    """
    threshold_band = threshold * abs(setpoint) if abs(setpoint) > 1e-6 else 0.01
    
    for i in range(len(t)-20):  # Check if position stays in band for at least 20 samples
        if abs(positions[i] - setpoint) <= threshold_band:
            # Check if position stays within band for next 20 samples
            if all(abs(positions[j] - setpoint) <= threshold_band for j in range(i, min(i+20, len(positions)))):
                return t[i]
    
    # If never settles, return the full simulation time as a penalty
    return t[-1]

def calculate_overshoot(positions, setpoint):
    """
    Calculate the percentage overshoot.
    
    Parameters:
    -----------
    positions : array
        Ball position array
    setpoint : float
        Desired ball position
        
    Returns:
    --------
    float
        Percentage overshoot
    """
    if abs(setpoint) < 1e-6:
        # Special case for zero setpoint
        max_abs_position = max(abs(p) for p in positions)
        return max_abs_position * 100 if max_abs_position > 0 else 0
    
    # Find first crossing of setpoint
    first_crossing_idx = None
    for i in range(len(positions)-1):
        if (positions[i] < setpoint and positions[i+1] >= setpoint) or \
           (positions[i] > setpoint and positions[i+1] <= setpoint):
            first_crossing_idx = i+1
            break
    
    if first_crossing_idx is None:
        # If position never crosses setpoint
        max_pos = max(positions)
        min_pos = min(positions)
        if setpoint > 0 and max_pos < setpoint:
            return 0  # Never reaches positive setpoint
        elif setpoint < 0 and min_pos > setpoint:
            return 0  # Never reaches negative setpoint
        elif setpoint == 0:
            return max(abs(max_pos), abs(min_pos)) * 100
    
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

def run_comprehensive_tests(fuzzy_controller, pid_controller):
    """
    Run a comprehensive suite of tests with different scenarios.
    
    Parameters:
    -----------
    fuzzy_controller : EnhancedFuzzyController
        Fuzzy controller to test
    pid_controller : OptimizedPIDController
        PID controller to test
        
    Returns:
    --------
    list
        List of test results
    """
    system = BallAndBeamSystem()
    test_cases = [
        # (name, initial_position, initial_velocity, setpoint)
        ("Standard Case", 0.0, 0.0, 0.5),
        ("Distant Start", -0.8, 0.0, 0.5),
        ("Initial Velocity", 0.0, 0.3, 0.5),
        ("Negative Setpoint", 0.0, 0.0, -0.3),
        ("Distant With Velocity", 0.7, -0.2, -0.2)
    ]
    
    results = []
    
    for name, init_pos, init_vel, setpoint in test_cases:
        logging.info(f"\n=== Test Case: {name} ===")
        
        # Run tests
        initial_state = [init_pos, init_vel]
        fuzzy_results = test_controller(system, fuzzy_controller, initial_state, setpoint, title=name)
        pid_results = test_controller(system, pid_controller, initial_state, setpoint, title=name)
        
        # Store results
        results.append({
            "name": name,
            "initial_state": initial_state,
            "setpoint": setpoint,
            "fuzzy": fuzzy_results,
            "pid": pid_results
        })
    
    # Generate summary table
    print_comparison_table(results)
    
    return results

def print_comparison_table(results):
    """
    Print a comparison table of controller performance.
    
    Parameters:
    -----------
    results : list
        List of test results
    """
    headers = ["Test Case", "Controller", "Settling Time (s)", "Overshoot (%)", "Steady-State Error (m)"]
    rows = []
    
    for result in results:
        name = result["name"]
        
        # Fuzzy controller metrics
        fuzzy_metrics = result["fuzzy"]["metrics"]
        rows.append([
            name,
            "Fuzzy",
            f"{fuzzy_metrics['settling_time']:.3f}",
            f"{fuzzy_metrics['overshoot']:.2f}",
            f"{fuzzy_metrics['steady_state_error']:.6f}"
        ])
        
        # PID controller metrics
        pid_metrics = result["pid"]["metrics"]
        rows.append([
            "",
            "PID",
            f"{pid_metrics['settling_time']:.3f}",
            f"{pid_metrics['overshoot']:.2f}",
            f"{pid_metrics['steady_state_error']:.6f}"
        ])
    
    # Calculate table column widths
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    
    # Print header
    header_str = " | ".join(f"{headers[i]:{col_widths[i]}}" for i in range(len(headers)))
    logging.info("\nPerformance Comparison:")
    logging.info(header_str)
    logging.info("-" * len(header_str))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(f"{row[i]:{col_widths[i]}}" for i in range(len(row)))
        logging.info(row_str)
    
    # Calculate averages
    fuzzy_settling = np.mean([result["fuzzy"]["metrics"]["settling_time"] for result in results])
    fuzzy_overshoot = np.mean([result["fuzzy"]["metrics"]["overshoot"] for result in results])
    fuzzy_error = np.mean([result["fuzzy"]["metrics"]["steady_state_error"] for result in results])
    
    pid_settling = np.mean([result["pid"]["metrics"]["settling_time"] for result in results])
    pid_overshoot = np.mean([result["pid"]["metrics"]["overshoot"] for result in results])
    pid_error = np.mean([result["pid"]["metrics"]["steady_state_error"] for result in results])
    
    logging.info("-" * len(header_str))
    logging.info(f"{'Average'} | {'Fuzzy'} | {fuzzy_settling:.3f} | {fuzzy_overshoot:.2f} | {fuzzy_error:.6f}")
    logging.info(f"{''} | {'PID'} | {pid_settling:.3f} | {pid_overshoot:.2f} | {pid_error:.6f}")

def create_demo_animation(result, save_path="ball_beam_demo.mp4"):
    """
    Create and save an animation of the ball and beam system.
    
    Parameters:
    -----------
    result : dict
        Simulation result
    save_path : str
        Path to save the animation
        
    Returns:
    --------
    FuncAnimation
        Animation object
    """
    time_data = result["time"]
    position_data = result["position"]
    angle_data = result["control"]
    setpoint = result.get("setpoint", 0.5)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Beam dimensions
    beam_length = 1.0
    beam_height = 0.05
    
    # Ball dimensions
    ball_radius = 0.04
    
    # Function to update animation frame
    def update(frame):
        ax.clear()
        
        # Set fixed axes
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.4, 0.4)
        
        # Current time, position and angle
        t = time_data[frame]
        pos = position_data[frame]
        angle = angle_data[frame]
        
        # Draw beam
        beam_x = [-beam_length/2, beam_length/2]
        beam_y = [np.sin(angle) * -beam_length/2, np.sin(angle) * beam_length/2]
        ax.plot(beam_x, beam_y, 'k-', linewidth=4)
        
        # Draw ball
        # Transform position from 0-1.0 range to -0.5 to 0.5 range on beam
        plot_pos = pos - 0.5  
        ball_x = plot_pos
        ball_y = np.sin(angle) * plot_pos + beam_height/2 + ball_radius
        ball = plt.Circle((ball_x, ball_y), ball_radius, color='blue')
        ax.add_patch(ball)
        
        # Draw setpoint marker
        setpoint_x = setpoint - 0.5  # Transform setpoint
        setpoint_y = np.sin(angle) * setpoint_x
        ax.plot(setpoint_x, setpoint_y, 'ro')
        
        # Add info text
        ax.text(0.02, 0.95, f"Time: {t:.2f}s", transform=ax.transAxes)
        ax.text(0.02, 0.90, f"Position: {pos:.3f}m", transform=ax.transAxes)
        ax.text(0.02, 0.85, f"Angle: {angle:.3f}rad", transform=ax.transAxes)
        ax.text(0.02, 0.80, f"Error: {setpoint - pos:.3f}m", transform=ax.transAxes)
        
        ax.set_title("Ball and Beam Control System")
        ax.set_aspect('equal')
        
    # Create animation - use fewer frames for smoother playback
    frames = min(len(time_data), 300)  # Limit to 300 frames
    frame_indices = np.linspace(0, len(time_data)-1, frames).astype(int)
    
    ani = FuncAnimation(fig, update, frames=frame_indices, interval=33.33, blit=False)
    
    # Save as video if possible
    try:
        ani.save(save_path, writer='ffmpeg', fps=30)
        logging.info(f"Animation saved to {save_path}")
    except Exception as e:
        logging.info(f"Could not save animation: {e}")
        logging.info("You may need to install ffmpeg to save animations.")
    
    plt.close()
    return ani

def create_side_by_side_demo(fuzzy_result, pid_result, save_path="comparison_demo.mp4"):
    """
    Create a side-by-side comparison demo of fuzzy and PID controllers.
    
    Parameters:
    -----------
    fuzzy_result : dict
        Fuzzy controller simulation result
    pid_result : dict
        PID controller simulation result
    save_path : str
        Path to save the animation
        
    Returns:
    --------
    FuncAnimation
        Animation object
    """
    time_data = fuzzy_result["time"]  # Assuming same time for both
    fuzzy_position = fuzzy_result["position"]
    fuzzy_angle = fuzzy_result["control"]
    pid_position = pid_result["position"]
    pid_angle = pid_result["control"]
    setpoint = fuzzy_result.get("setpoint", 0.5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Beam dimensions
    beam_length = 1.0
    beam_height = 0.05
    
    # Ball dimensions
    ball_radius = 0.04
    
    # Function to update animation frame
    def update(frame):
        for ax, position, angle, title in [
            (ax1, fuzzy_position, fuzzy_angle, "Fuzzy Controller"),
            (ax2, pid_position, pid_angle, "PID Controller")
        ]:
            ax.clear()
            
            # Set fixed axes
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(-0.4, 0.4)
            
            # Current time, position and angle
            t = time_data[frame]
            pos = position[frame]
            ang = angle[frame]
            
            # Draw beam
            beam_x = [-beam_length/2, beam_length/2]
            beam_y = [np.sin(ang) * -beam_length/2, np.sin(ang) * beam_length/2]
            ax.plot(beam_x, beam_y, 'k-', linewidth=4)
            
            # Draw ball
            # Transform position from 0-1.0 range to -0.5 to 0.5 range on beam
            plot_pos = pos - 0.5  
            ball_x = plot_pos
            ball_y = np.sin(ang) * plot_pos + beam_height/2 + ball_radius
            ball = plt.Circle((ball_x, ball_y), ball_radius, color='blue')
            ax.add_patch(ball)
            
            # Draw setpoint marker
            setpoint_x = setpoint - 0.5  # Transform setpoint
            setpoint_y = np.sin(ang) * setpoint_x
            ax.plot(setpoint_x, setpoint_y, 'ro')
            
            # Add info text
            ax.text(0.02, 0.95, f"Time: {t:.2f}s", transform=ax.transAxes)
            ax.text(0.02, 0.90, f"Position: {pos:.3f}m", transform=ax.transAxes)
            ax.text(0.02, 0.85, f"Error: {setpoint - pos:.3f}m", transform=ax.transAxes)
            
            ax.set_title(title)
            ax.set_aspect('equal')
        
        fig.suptitle("Ball and Beam Control System Comparison", fontsize=16)
        
    # Create animation - use fewer frames for smoother playback
    frames = min(len(time_data), 300)  # Limit to 300 frames
    frame_indices = np.linspace(0, len(time_data)-1, frames).astype(int)
    
    ani = FuncAnimation(fig, update, frames=frame_indices, interval=33.33, blit=False)
    
    # Save as video if possible
    try:
        ani.save(save_path, writer='ffmpeg', fps=30)
        logging.info(f"Comparison animation saved to {save_path}")
    except Exception as e:
        logging.info(f"Could not save animation: {e}")
        logging.info("You may need to install ffmpeg to save animations.")
    
    plt.close()
    return ani

def save_results_to_csv(results, filename="controller_results.csv"):
    """
    Save test results to a CSV file.
    
    Parameters:
    -----------
    results : list
        List of test results
    filename : str
        Filename to save results
    """
    data = []
    
    for result in results:
        name = result["name"]
        initial_pos, initial_vel = result["initial_state"]
        setpoint = result["setpoint"]
        
        # Fuzzy controller metrics
        fuzzy_metrics = result["fuzzy"]["metrics"]
        data.append({
            "Test Case": name,
            "Initial Position": initial_pos,
            "Initial Velocity": initial_vel,
            "Setpoint": setpoint,
            "Controller": "Fuzzy",
            "Settling Time": fuzzy_metrics["settling_time"],
            "Overshoot": fuzzy_metrics["overshoot"],
            "Steady-State Error": fuzzy_metrics["steady_state_error"]
        })
        
        # PID controller metrics
        pid_metrics = result["pid"]["metrics"]
        data.append({
            "Test Case": name,
            "Initial Position": initial_pos,
            "Initial Velocity": initial_vel,
            "Setpoint": setpoint,
            "Controller": "PID",
            "Settling Time": pid_metrics["settling_time"],
            "Overshoot": pid_metrics["overshoot"],
            "Steady-State Error": pid_metrics["steady_state_error"]
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logging.info(f"Results saved to {filename}")

def main():
    """Main function to run the ball and beam control system."""
    # Create output directory for results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{results_dir}/simulation_log_{timestamp}.log"

    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting ball and beam simulation")
    
    # Create controllers
    # fuzzy_controller = EnhancedFuzzyController()
    fuzzy_controller = ImprovedFuzzyController()
    pid_controller = OptimizedPIDController()
    
    # Run comprehensive tests
    results = run_comprehensive_tests(fuzzy_controller, pid_controller)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results_to_csv(results, f"{results_dir}/controller_results_{timestamp}.csv")
    
    # Create demo animation for the standard case (first test case)
    std_case = results[0]
    create_demo_animation(
        std_case["fuzzy"], 
        f"{results_dir}/fuzzy_demo_{timestamp}.mp4"
    )
    
    # Create side-by-side comparison demo
    create_side_by_side_demo(
        std_case["fuzzy"],
        std_case["pid"],
        f"{results_dir}/comparison_demo_{timestamp}.mp4"
    )
    
    logging.info("\nTesting complete. Results and animations saved to the 'results' directory.")

if __name__ == "__main__":
    main()

