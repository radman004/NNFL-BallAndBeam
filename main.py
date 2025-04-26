"""
Main application for the ball and beam control system.
"""
import numpy as np
import pygame
import time

# Import project modules
from physics import BallAndBeamSystem
from fuzzy_controller import FuzzyController
from pid_controller import PIDController
from visualization import BallAndBeamVisualization
from performance import PerformanceAnalyzer

def main():
    """Main function to run the ball and beam control system."""
    # Initialize pygame
    pygame.init()
    
    # Initialize systems
    beam_length = 1.0
    max_angle_deg = 15
    
    # Create the ball and beam systems
    fuzzy_system = BallAndBeamSystem(beam_length=beam_length, max_angle_deg=max_angle_deg)
    pid_system = BallAndBeamSystem(beam_length=beam_length, max_angle_deg=max_angle_deg)
    
    # Create controllers
    fuzzy_controller = FuzzyController(beam_length=beam_length, max_angle_deg=max_angle_deg)
    pid_controller = PIDController(Kp=5.0, Ki=0.0, Kd=2.0, max_angle_deg=max_angle_deg)
    
    # Initialize visualization
    viz = BallAndBeamVisualization()
    
    # Initialize performance analyzers
    fuzzy_analyzer = PerformanceAnalyzer()
    pid_analyzer = PerformanceAnalyzer()
    
    # Set initial conditions
    initial_position = -0.3
    fuzzy_system.reset(position=initial_position)
    pid_system.reset(position=initial_position)
    pid_controller.reset()
    
    # Target position
    target_position = 0.2
    
    # Simulation parameters
    dt = 0.01  # time step (seconds)
    sim_time = 0.0  # current simulation time
    running = True
    paused = False
    
    # Main simulation loop
    clock = pygame.time.Clock()
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Reset simulation
                    fuzzy_system.reset(position=initial_position)
                    pid_system.reset(position=initial_position)
                    pid_controller.reset()
                    fuzzy_analyzer.reset()
                    pid_analyzer.reset()
                    sim_time = 0.0
                    viz.start_time = None  # Reset convergence tracking
                    viz.converged = {'fuzzy': False, 'pid': False}
                    viz.convergence_times = {'fuzzy': None, 'pid': None}
        
        if not paused:
            # Compute control inputs
            fuzzy_error = target_position - fuzzy_system.position
            fuzzy_angle = fuzzy_controller.compute(fuzzy_error, fuzzy_system.velocity)
            
            pid_error = target_position - pid_system.position
            pid_angle = pid_controller.compute(pid_error, pid_system.velocity, dt)
            
            # Update systems
            fuzzy_system.update(np.radians(fuzzy_angle), dt)
            pid_system.update(np.radians(pid_angle), dt)
            
            # Update performance analyzers
            fuzzy_metrics = fuzzy_analyzer.update(sim_time, fuzzy_system.position, target_position)
            pid_metrics = pid_analyzer.update(sim_time, pid_system.position, target_position)
            
            # Update visualization
            viz.update(fuzzy_system, pid_system, target_position, sim_time)
            
            # Increment time
            sim_time += dt
        
        # Cap the frame rate
        clock.tick(100)
    
    # Clean up
    pygame.quit()
    
    # Print final performance metrics
    print("\nPerformance Metrics Summary:")
    print("\nFuzzy Logic Controller:")
    print(f"  Max Overshoot: {fuzzy_metrics['max_overshoot']:.2f}%")
    print(f"  Settling Time: {fuzzy_metrics['settling_time']:.2f} s")
    print(f"  Steady-State Error: {fuzzy_metrics['steady_state_error']:.5f} m")
    print(f"  IAE: {fuzzy_metrics['integral_absolute_error']:.5f}")
    
    print("\nPID Controller:")
    print(f"  Max Overshoot: {pid_metrics['max_overshoot']:.2f}%")
    print(f"  Settling Time: {pid_metrics['settling_time']:.2f} s")
    print(f"  Steady-State Error: {pid_metrics['steady_state_error']:.5f} m")
    print(f"  IAE: {pid_metrics['integral_absolute_error']:.5f}")
    
    if viz.converged['fuzzy'] and viz.converged['pid']:
        faster = "Fuzzy" if viz.convergence_times['fuzzy'] < viz.convergence_times['pid'] else "PID"
        diff = abs(viz.convergence_times['fuzzy'] - viz.convergence_times['pid'])
        print(f"\n{faster} controller converged {diff:.2f}s faster")

if __name__ == "__main__":
    main()