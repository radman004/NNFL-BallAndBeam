import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.integrate import solve_ivp
import time






class BallAndBeamSystem:
    def __init__(self, m=0.1, r=0.015, g=9.81, J=9.99e-6, b=0.01, L=1.0):
        """
        Initialize the ball and beam system parameters
        
        Parameters:
        -----------
        m : float
            Mass of the ball in kg
        r : float
            Radius of the ball in m
        g : float
            Gravitational acceleration in m/s^2
        J : float
            Moment of inertia of the ball in kg*m^2
        b : float
            Friction coefficient
        L : float
            Length of the beam in m
        """
        self.m = m          # Mass of ball
        self.r = r          # Radius of ball
        self.g = g          # Gravitational acceleration
        self.J = J          # Moment of inertia of ball
        self.b = b          # Friction coefficient
        self.L = L          # Length of beam
        
    def dynamics(self, t, state, alpha):
        """
        Calculate the derivatives of the state variables
        
        Parameters:
        -----------
        t : float
            Time (not used, but required by solve_ivp)
        state : array-like
            Current state [position, velocity]
        alpha : float
            Beam angle in radians
            
        Returns:
        --------
        array-like
            Derivatives of the state [velocity, acceleration]
        """
        x, x_dot = state
        
        # The ball and beam dynamics equation
        x_dotdot = (self.m * self.g * np.sin(alpha) - self.b * x_dot) / (self.J/self.r**2 + self.m)
        
        return [x_dot, x_dotdot]
    
class FuzzyController:
    def __init__(self, universe_range=(-1, 1), beam_angle_limit=0.5):
        """
        Initialize the fuzzy logic controller with 7 terms
        
        Parameters:
        -----------
        universe_range : tuple
            Range for input universes (error, error_derivative)
        beam_angle_limit : float
            Maximum beam angle in radians
        """
        # Define the universes
        error_range = np.linspace(universe_range[0], universe_range[1], 100)
        derror_range = np.linspace(universe_range[0], universe_range[1], 100)
        output_range = np.linspace(-beam_angle_limit, beam_angle_limit, 100)
        
        # Create the fuzzy variables
        error = ctrl.Antecedent(error_range, 'error')
        derror = ctrl.Antecedent(derror_range, 'derror')
        output = ctrl.Consequent(output_range, 'output')
        
        # Define membership functions with 7 terms
        names = ['NL', 'NM', 'NS', 'Z', 'PS', 'PM', 'PL']
        
        # Auto-generate membership functions
        error.automf(names=names)
        derror.automf(names=names)
        output.automf(names=names)
        
        # Define the rule base (7x7 matrix)
        rule1 = ctrl.Rule(error['NL'] & derror['NL'], output['NL'])
        rule2 = ctrl.Rule(error['NL'] & derror['NM'], output['NL'])
        rule3 = ctrl.Rule(error['NL'] & derror['NS'], output['NL'])
        rule4 = ctrl.Rule(error['NL'] & derror['Z'], output['NM'])
        rule5 = ctrl.Rule(error['NL'] & derror['PS'], output['NS'])
        rule6 = ctrl.Rule(error['NL'] & derror['PM'], output['NS'])
        rule7 = ctrl.Rule(error['NL'] & derror['PL'], output['Z'])
        
        rule8 = ctrl.Rule(error['NM'] & derror['NL'], output['NL'])
        rule9 = ctrl.Rule(error['NM'] & derror['NM'], output['NL'])
        rule10 = ctrl.Rule(error['NM'] & derror['NS'], output['NM'])
        rule11 = ctrl.Rule(error['NM'] & derror['Z'], output['NM'])
        rule12 = ctrl.Rule(error['NM'] & derror['PS'], output['NS'])
        rule13 = ctrl.Rule(error['NM'] & derror['PM'], output['Z'])
        rule14 = ctrl.Rule(error['NM'] & derror['PL'], output['PS'])
        
        rule15 = ctrl.Rule(error['NS'] & derror['NL'], output['NL'])
        rule16 = ctrl.Rule(error['NS'] & derror['NM'], output['NM'])
        rule17 = ctrl.Rule(error['NS'] & derror['NS'], output['NS'])
        rule18 = ctrl.Rule(error['NS'] & derror['Z'], output['NS'])
        rule19 = ctrl.Rule(error['NS'] & derror['PS'], output['Z'])
        rule20 = ctrl.Rule(error['NS'] & derror['PM'], output['PS'])
        rule21 = ctrl.Rule(error['NS'] & derror['PL'], output['PM'])
        
        rule22 = ctrl.Rule(error['Z'] & derror['NL'], output['NM'])
        rule23 = ctrl.Rule(error['Z'] & derror['NM'], output['NM'])
        rule24 = ctrl.Rule(error['Z'] & derror['NS'], output['NS'])
        rule25 = ctrl.Rule(error['Z'] & derror['Z'], output['Z'])
        rule26 = ctrl.Rule(error['Z'] & derror['PS'], output['PS'])
        rule27 = ctrl.Rule(error['Z'] & derror['PM'], output['PM'])
        rule28 = ctrl.Rule(error['Z'] & derror['PL'], output['PM'])
        
        rule29 = ctrl.Rule(error['PS'] & derror['NL'], output['NM'])
        rule30 = ctrl.Rule(error['PS'] & derror['NM'], output['NS'])
        rule31 = ctrl.Rule(error['PS'] & derror['NS'], output['Z'])
        rule32 = ctrl.Rule(error['PS'] & derror['Z'], output['PS'])
        rule33 = ctrl.Rule(error['PS'] & derror['PS'], output['PS'])
        rule34 = ctrl.Rule(error['PS'] & derror['PM'], output['PM'])
        rule35 = ctrl.Rule(error['PS'] & derror['PL'], output['PL'])
        
        rule36 = ctrl.Rule(error['PM'] & derror['NL'], output['NS'])
        rule37 = ctrl.Rule(error['PM'] & derror['NM'], output['Z'])
        rule38 = ctrl.Rule(error['PM'] & derror['NS'], output['PS'])
        rule39 = ctrl.Rule(error['PM'] & derror['Z'], output['PM'])
        rule40 = ctrl.Rule(error['PM'] & derror['PS'], output['PM'])
        rule41 = ctrl.Rule(error['PM'] & derror['PM'], output['PL'])
        rule42 = ctrl.Rule(error['PM'] & derror['PL'], output['PL'])
        
        rule43 = ctrl.Rule(error['PL'] & derror['NL'], output['Z'])
        rule44 = ctrl.Rule(error['PL'] & derror['NM'], output['PS'])
        rule45 = ctrl.Rule(error['PL'] & derror['NS'], output['PS'])
        rule46 = ctrl.Rule(error['PL'] & derror['Z'], output['PM'])
        rule47 = ctrl.Rule(error['PL'] & derror['PS'], output['PL'])
        rule48 = ctrl.Rule(error['PL'] & derror['PM'], output['PL'])
        rule49 = ctrl.Rule(error['PL'] & derror['PL'], output['PL'])
        
        # Consolidate all rules
        rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7,
                rule8, rule9, rule10, rule11, rule12, rule13, rule14,
                rule15, rule16, rule17, rule18, rule19, rule20, rule21,
                rule22, rule23, rule24, rule25, rule26, rule27, rule28,
                rule29, rule30, rule31, rule32, rule33, rule34, rule35,
                rule36, rule37, rule38, rule39, rule40, rule41, rule42,
                rule43, rule44, rule45, rule46, rule47, rule48, rule49]
        
        # Create the control system
        self.control_system = ctrl.ControlSystem(rules)
        self.controller = ctrl.ControlSystemSimulation(self.control_system)
        
    def compute(self, error, derror):
        """
        Compute the control output based on error and error derivative
        
        Parameters:
        -----------
        error : float
            Position error
        derror : float
            Derivative of position error
            
        Returns:
        --------
        float
            Control output (beam angle)
        """
        # Clip inputs to universe ranges to prevent errors
        error = np.clip(error, -1, 1)
        derror = np.clip(derror, -1, 1)
        
        # Set inputs
        self.controller.input['error'] = error
        self.controller.input['derror'] = derror
        
        # Compute output
        try:
            self.controller.compute()
            return self.controller.output['output']
        except:
            # Fallback in case of computational error
            print("Fuzzy computation error!")
            return 0
        
class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0.0, beam_angle_limit=0.5):
        """
        Initialize the PID controller
        
        Parameters:
        -----------
        Kp : float
            Proportional gain
        Ki : float
            Integral gain
        Kd : float
            Derivative gain
        setpoint : float
            Desired position
        beam_angle_limit : float
            Maximum beam angle in radians
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.beam_angle_limit = beam_angle_limit
        self.error_sum = 0.0
        self.last_error = 0.0
        self.last_time = None
        
    def compute(self, position, current_time=None):
        """
        Compute the control output based on current position
        
        Parameters:
        -----------
        position : float
            Current ball position
        current_time : float
            Current time (for dt calculation)
            
        Returns:
        --------
        float
            Control output (beam angle)
        """
        # Calculate current error
        error = self.setpoint - position
        
        # Handle time for derivative and integral terms
        if current_time is None:
            current_time = time.time()
        
        if self.last_time is None:
            dt = 0.01  # Default dt on first call
        else:
            dt = current_time - self.last_time
            
        # Ensure dt is not too small to avoid division issues
        if dt < 1e-6:
            dt = 1e-6
            
        # Calculate PID terms
        P_term = self.Kp * error
        
        # Integral term with anti-windup
        self.error_sum += error * dt
        I_term = self.Ki * self.error_sum
        
        # Derivative term (on error, not measurement)
        error_deriv = (error - self.last_error) / dt
        D_term = self.Kd * error_deriv
        
        # Total control output
        output = P_term + I_term + D_term
        
        # Clip output to beam angle limits
        output = np.clip(output, -self.beam_angle_limit, self.beam_angle_limit)
        
        # Save values for next iteration
        self.last_error = error
        self.last_time = current_time
        
        return output        

class Simulator:
    def __init__(self, system, controller, setpoint=0.4, dt=0.01, t_max=10.0, initial_state=None):
        """
        Initialize the simulator
        
        Parameters:
        -----------
        system : BallAndBeamSystem
            The ball and beam system to simulate
        controller : FuzzyController or PIDController
            The controller to use
        setpoint : float
            Desired position for the ball
        dt : float
            Time step for simulation
        t_max : float
            Maximum simulation time
        initial_state : array-like
            Initial state [position, velocity]
        """
        self.system = system
        self.controller = controller
        self.setpoint = setpoint
        self.dt = dt
        self.t_max = t_max
        
        if initial_state is None:
            self.initial_state = [0.0, 0.0]  # Default: ball at center, not moving
        else:
            self.initial_state = initial_state
            
        # Set up for data collection
        self.time_points = np.arange(0, t_max, dt)
        self.num_points = len(self.time_points)
        self.positions = np.zeros(self.num_points)
        self.velocities = np.zeros(self.num_points)
        self.angles = np.zeros(self.num_points)
        
    def run(self):
        """
        Run the simulation
        
        Returns:
        --------
        dict
            Simulation results
        """
        # Initialize state
        state = self.initial_state.copy()
        
        # For PID controller
        if hasattr(self.controller, 'setpoint'):
            self.controller.setpoint = self.setpoint
            self.controller.error_sum = 0.0
            self.controller.last_error = 0.0
            self.controller.last_time = None
        
        # Run simulation
        for i, t in enumerate(self.time_points):
            # Save current state
            self.positions[i] = state[0]
            self.velocities[i] = state[1]
            
            # Calculate error and its derivative
            error = self.setpoint - state[0]
            derror = -state[1]  # Negative velocity is equivalent to positive error derivative
            
            # Compute control input
            if hasattr(self.controller, 'setpoint'):  # PID controller
                beam_angle = self.controller.compute(state[0], t)
            else:  # Fuzzy controller
                beam_angle = self.controller.compute(error, derror)
            
            self.angles[i] = beam_angle
            
            # Integrate system dynamics for one time step
            sol = solve_ivp(
                lambda t, y: self.system.dynamics(t, y, beam_angle),
                [0, self.dt],
                state,
                method='RK45',
                t_eval=[self.dt]
            )
            
            # Update state
            state = sol.y[:, -1]
        
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        
        # Return results
        results = {
            'time': self.time_points,
            'position': self.positions,
            'velocity': self.velocities,
            'angle': self.angles,
            'setpoint': self.setpoint,
            'metrics': metrics
        }
        
        return results
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
        --------
        dict
            Performance metrics
        """
        # Find steady state (last 10% of simulation)
        steady_idx = int(0.9 * self.num_points)
        steady_positions = self.positions[steady_idx:]
        
        # Calculate steady-state error
        steady_error = np.abs(self.setpoint - np.mean(steady_positions))
        steady_error_percent = steady_error / self.setpoint * 100 if self.setpoint != 0 else 0
        
        # Calculate overshoot
        if self.initial_state[0] < self.setpoint:
            overshoot = max(0, np.max(self.positions) - self.setpoint)
        else:
            overshoot = max(0, self.setpoint - np.min(self.positions))
            
        overshoot_percent = overshoot / np.abs(self.setpoint - self.initial_state[0]) * 100 if self.setpoint != self.initial_state[0] else 0
        
        # Calculate settling time (within 2% of setpoint)
        tolerance = 0.02 * np.abs(self.setpoint)
        settled = np.where(np.abs(self.positions - self.setpoint) <= tolerance)[0]
        
        if len(settled) > 0:
            # Find the last time the response enters the tolerance band
            last_entry = settled[0]
            for i in range(1, len(settled)):
                if settled[i] > settled[i-1] + 1:  # Gap in settled points
                    last_entry = settled[i]
            
            settling_time = self.time_points[last_entry]
        else:
            settling_time = np.inf
        
        metrics = {
            'steady_state_error': steady_error,
            'steady_state_error_percent': steady_error_percent,
            'overshoot': overshoot,
            'overshoot_percent': overshoot_percent,
            'settling_time': settling_time
        }
        
        return metrics

def plot_results(results, controller_name):
    """
    Plot simulation results
    
    Parameters:
    -----------
    results : dict
        Simulation results
    controller_name : str
        Name of the controller
    """
    time = results['time']
    position = results['position']
    angle = results['angle']
    setpoint = results['setpoint']
    metrics = results['metrics']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot ball position
    ax1.plot(time, position, label='Ball Position')
    ax1.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title(f'{controller_name} - Ball Position')
    ax1.grid(True)
    ax1.legend()
    
    # Plot beam angle
    ax2.plot(time, angle, label='Beam Angle')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title(f'{controller_name} - Beam Angle')
    ax2.grid(True)
    ax2.legend()
    
    # Add metrics as text
    metrics_text = (
        f"Overshoot: {metrics['overshoot_percent']:.2f}%\n"
        f"Steady-state Error: {metrics['steady_state_error_percent']:.2f}%\n"
        f"Settling Time: {metrics['settling_time']:.2f} s"
    )
    
    plt.figtext(0.02, 0.02, metrics_text, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{controller_name.lower().replace(" ", "_")}_results.png')
    plt.show()

def animate_system(results, system, controller_name, save_animation=False):
    """
    Create an animation of the ball and beam system
    
    Parameters:
    -----------
    results : dict
        Simulation results
    system : BallAndBeamSystem
        The ball and beam system
    controller_name : str
        Name of the controller
    save_animation : bool
        Whether to save the animation as a GIF
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-system.L/2, system.L/2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Create beam and ball objects
    beam, = ax.plot([], [], 'k-', lw=3)
    ball, = ax.plot([], [], 'ro', markersize=10)
    
    # Title with controller name
    ax.set_title(f'Ball and Beam System - {controller_name}')
    
    # Text for displaying metrics
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    position_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    angle_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    
    def init():
        beam.set_data([], [])
        ball.set_data([], [])
        time_text.set_text('')
        position_text.set_text('')
        angle_text.set_text('')
        return beam, ball, time_text, position_text, angle_text
    
    def animate(i):
        # Get data for current frame
        t = results['time'][i]
        x = results['position'][i]
        alpha = results['angle'][i]
        
        # Calculate beam endpoints
        x1, y1 = -system.L/2, -np.sin(alpha) * system.L/2
        x2, y2 = system.L/2, np.sin(alpha) * system.L/2
        
        # Update beam
        beam.set_data([x1, x2], [y1, y2])
        
        # Calculate ball position
        ball_x = x - system.L/2
        # Height of the beam at position x
        ball_y = y1 + (ball_x - x1) * (y2 - y1) / (x2 - x1)
        
        # Update ball
        ball.set_data([ball_x], [ball_y])
        
        # Update text
        time_text.set_text(f'Time: {t:.2f} s')
        position_text.set_text(f'Position: {x:.2f} m')
        angle_text.set_text(f'Angle: {alpha*180/np.pi:.2f} deg')
        
        return beam, ball, time_text, position_text, angle_text
    
    # Create animation
    frames = min(len(results['time']), 200)  # Limit frames for performance
    step = max(1, len(results['time']) // frames)
    
    anim = FuncAnimation(fig, animate, frames=range(0, len(results['time']), step),
                         init_func=init, blit=True, interval=50)
    
    if save_animation:
        anim.save(f'{controller_name.lower().replace(" ", "_")}_animation.gif', writer='pillow', fps=20)
    
    plt.show()


def main():
    # Create system
    system = BallAndBeamSystem()
    
    # Create controllers
    fuzzy_controller = FuzzyController()
    # Starting with PID values from literature, will need tuning
    pid_controller = PIDController(Kp=2.0, Ki=0.1, Kd=1.0)
    
    # Define simulation parameters
    setpoint = 0.4  # Target position (m)
    initial_state = [0.0, 0.0]  # Start at center with no velocity
    sim_time = 10.0  # Simulation time (s)
    
    # Run simulations
    print("Running Fuzzy Logic Controller simulation...")
    fuzzy_simulator = Simulator(system, fuzzy_controller, setpoint, 
                              initial_state=initial_state, t_max=sim_time)
    fuzzy_results = fuzzy_simulator.run()
    
    print("Running PID Controller simulation...")
    pid_simulator = Simulator(system, pid_controller, setpoint, 
                            initial_state=initial_state, t_max=sim_time)
    pid_results = pid_simulator.run()
    
    # Plot results
    plot_results(fuzzy_results, "Fuzzy Logic Controller")
    plot_results(pid_results, "PID Controller")
    
    # Print comparison
    print("\nPerformance Comparison:")
    print("=" * 50)
    print(f"{'Metric':<20} {'Fuzzy Logic':<15} {'PID':<15}")
    print("-" * 50)
    
    metrics = [
        ('Overshoot (%)', 'overshoot_percent'),
        ('Steady Error (%)', 'steady_state_error_percent'),
        ('Settling Time (s)', 'settling_time')
    ]
    
    for name, key in metrics:
        fuzzy_value = fuzzy_results['metrics'][key]
        pid_value = pid_results['metrics'][key]
        print(f"{name:<20} {fuzzy_value:<15.2f} {pid_value:<15.2f}")
    
    # Create animations
    animate_system(fuzzy_results, system, "Fuzzy Logic Controller", save_animation=True)
    animate_system(pid_results, system, "PID Controller", save_animation=True)

def plot_combined_results(fuzzy_results, pid_results):
    """
    Plot fuzzy logic and PID controller results on the same axes for comparison
    """
    # Extract data
    time_fuzzy = fuzzy_results['time']
    position_fuzzy = fuzzy_results['position']
    angle_fuzzy = fuzzy_results['angle']
    
    time_pid = pid_results['time']
    position_pid = pid_results['position']
    angle_pid = pid_results['angle']
    
    setpoint = fuzzy_results['setpoint']  # Same setpoint for both
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    
    # Plot ball positions
    ax1.plot(time_fuzzy, position_fuzzy, 'b-', linewidth=2, label='Fuzzy Logic')
    ax1.plot(time_pid, position_pid, 'r-', linewidth=2, label='PID')
    ax1.axhline(y=setpoint, color='k', linestyle='--', label='Setpoint')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Ball Position Comparison')
    ax1.grid(True)
    ax1.legend()
    
    # Plot beam angles
    ax2.plot(time_fuzzy, angle_fuzzy, 'b-', linewidth=2, label='Fuzzy Logic')
    ax2.plot(time_pid, angle_pid, 'r-', linewidth=2, label='PID')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title('Beam Angle Comparison')
    ax2.grid(True)
    ax2.legend()
    
    # Add performance metrics as text
    fuzzy_metrics = fuzzy_results['metrics']
    pid_metrics = pid_results['metrics']
    
    metrics_text = (
        f"Fuzzy Logic:\n"
        f"  Overshoot: {fuzzy_metrics['overshoot_percent']:.2f}%\n"
        f"  Steady-state Error: {fuzzy_metrics['steady_state_error_percent']:.2f}%\n"
        f"  Settling Time: {fuzzy_metrics['settling_time']:.2f} s\n\n"
        f"PID:\n"
        f"  Overshoot: {pid_metrics['overshoot_percent']:.2f}%\n"
        f"  Steady-state Error: {pid_metrics['steady_state_error_percent']:.2f}%\n"
        f"  Settling Time: {pid_metrics['settling_time']:.2f} s"
    )
    
    plt.figtext(0.02, 0.02, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make room for metrics text
    plt.savefig('controller_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()



