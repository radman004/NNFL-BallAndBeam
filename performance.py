"""
Performance metrics for evaluating controller performance.
"""
class PerformanceAnalyzer:
    def __init__(self):
        """Initialize the performance analyzer."""
        self.reset()
        
    def reset(self):
        """Reset performance metrics."""
        self.metrics = {
            'max_overshoot': 0.0,
            'settling_time': float('inf'),
            'rise_time': float('inf'),
            'steady_state_error': 0.0,
            'integral_absolute_error': 0.0
        }
        self.settled = False
    
    def update(self, time, position, target):
        """Update performance metrics based on current state."""
        error = target - position
        abs_error = abs(error)
        
        # Update IAE
        self.metrics['integral_absolute_error'] += abs_error * 0.01  # Assuming dt = 0.01
        
        # Update overshoot
        if target > 0 and position > target:
            overshoot = (position - target) / target * 100
            self.metrics['max_overshoot'] = max(self.metrics['max_overshoot'], overshoot)
        elif target < 0 and position < target:
            overshoot = (target - position) / abs(target) * 100
            self.metrics['max_overshoot'] = max(self.metrics['max_overshoot'], overshoot)
        
        # Update settling time (if within 2% of target)
        if abs_error <= 0.02 * abs(target) and not self.settled:
            self.metrics['settling_time'] = time
            self.settled = True
        
        # Update steady-state error
        self.metrics['steady_state_error'] = abs_error
        
        return self.metrics