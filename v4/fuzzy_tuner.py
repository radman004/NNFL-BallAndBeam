# fuzzy_tuner.py
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np
import pyswarms as ps
from tqdm import tqdm
from bbs import BallAndBeamSystem, EnhancedFuzzyController, compute_metrics


class TunableFuzzyController(EnhancedFuzzyController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Redefine universes for tuning
        self.error = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'error')
        self.d_error = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'delta_error')
        self.angle = ctrl.Consequent(np.linspace(-0.4, 0.4, 1000), 'angle')

    def set_membership_params(self, params):
        # Unpack parameters
        (e_NL_left, e_NL_peak, e_ZE_peak, e_PL_peak, e_PL_right,
         de_NL_left, de_NL_peak, de_ZE_peak, de_PL_peak, de_PL_right,
         o_NL_left, o_NL_peak, o_ZE_peak, o_PL_peak, o_PL_right) = params

        # Ensure boundaries for universes
        e_NL_left = min(e_NL_left, -1.0)
        e_PL_right = max(e_PL_right, 1.0)
        de_NL_left = min(de_NL_left, -1.0)
        de_PL_right = max(de_PL_right, 1.0)
        o_NL_left = min(o_NL_left, -0.4)
        o_PL_right = max(o_PL_right, 0.4)

        # Error membership functions
        self.error['NB'] = fuzz.trimf(self.error.universe, sorted([e_NL_left, e_NL_left, e_NL_peak]))
        self.error['NM'] = fuzz.trimf(self.error.universe, sorted([e_NL_left, e_NL_peak, e_ZE_peak]))
        self.error['NS'] = fuzz.trimf(self.error.universe, sorted([e_NL_peak, e_ZE_peak, 0.0]))
        self.error['ZE'] = fuzz.trimf(self.error.universe, sorted([e_NL_peak, e_ZE_peak, e_PL_peak]))
        self.error['PS'] = fuzz.trimf(self.error.universe, sorted([0.0, e_ZE_peak, e_PL_peak]))
        self.error['PM'] = fuzz.trimf(self.error.universe, sorted([e_ZE_peak, e_PL_peak, e_PL_right]))
        self.error['PB'] = fuzz.trimf(self.error.universe, sorted([e_PL_peak, e_PL_right, e_PL_right]))

        # Delta-error membership functions
        self.d_error['NB'] = fuzz.trimf(self.d_error.universe, sorted([de_NL_left, de_NL_left, de_NL_peak]))
        self.d_error['NM'] = fuzz.trimf(self.d_error.universe, sorted([de_NL_left, de_NL_peak, de_ZE_peak]))
        self.d_error['NS'] = fuzz.trimf(self.d_error.universe, sorted([de_NL_peak, de_ZE_peak, 0.0]))
        self.d_error['ZE'] = fuzz.trimf(self.d_error.universe, sorted([de_NL_peak, de_ZE_peak, de_PL_peak]))
        self.d_error['PS'] = fuzz.trimf(self.d_error.universe, sorted([0.0, de_ZE_peak, de_PL_peak]))
        self.d_error['PM'] = fuzz.trimf(self.d_error.universe, sorted([de_ZE_peak, de_PL_peak, de_PL_right]))
        self.d_error['PB'] = fuzz.trimf(self.d_error.universe, sorted([de_PL_peak, de_PL_right, de_PL_right]))

        # Output membership functions
        self.angle['NB'] = fuzz.trimf(self.angle.universe, sorted([o_NL_left, o_NL_left, o_NL_peak]))
        self.angle['NM'] = fuzz.trimf(self.angle.universe, sorted([o_NL_left, o_NL_peak, o_ZE_peak]))
        self.angle['NS'] = fuzz.trimf(self.angle.universe, sorted([o_NL_peak, o_ZE_peak, 0.0]))
        self.angle['ZE'] = fuzz.trimf(self.angle.universe, sorted([o_NL_peak, o_ZE_peak, o_PL_peak]))
        self.angle['PS'] = fuzz.trimf(self.angle.universe, sorted([0.0, o_ZE_peak, o_PL_peak]))
        self.angle['PM'] = fuzz.trimf(self.angle.universe, sorted([o_ZE_peak, o_PL_peak, o_PL_right]))
        self.angle['PB'] = fuzz.trimf(self.angle.universe, sorted([o_PL_peak, o_PL_right, o_PL_right]))

        # Rebuild the fuzzy control system with updated MFs
        self._build_control_system()

    def _build_control_system(self):
        # Define fuzzy rules (reuse your existing rule set here)
        rules = [
            # NB (Ball far to right of setpoint)
            ctrl.Rule(self.error['NB'] & self.d_error['NB'], self.angle['NB']),  # Moving away fast → Strong negative
            ctrl.Rule(self.error['NB'] & self.d_error['NM'], self.angle['NB']),  # Moving away medium → Strong negative
            ctrl.Rule(self.error['NB'] & self.d_error['NS'], self.angle['NB']),  # Moving away slow → Strong negative
            ctrl.Rule(self.error['NB'] & self.d_error['ZE'], self.angle['NB']),  # Not moving → Strong negative
            ctrl.Rule(self.error['NB'] & self.d_error['PS'], self.angle['NM']),  # Moving toward slow → Medium negative
            ctrl.Rule(self.error['NB'] & self.d_error['PM'], self.angle['NS']),  # Moving toward medium → Small negative
            ctrl.Rule(self.error['NB'] & self.d_error['PB'], self.angle['ZE']),  # Moving toward fast → Zero angle
            
            # NM (Ball moderately right of setpoint)
            ctrl.Rule(self.error['NM'] & self.d_error['NB'], self.angle['NB']),  # Moving away fast → Strong negative
            ctrl.Rule(self.error['NM'] & self.d_error['NM'], self.angle['NB']),  # Moving away medium → Strong negative
            ctrl.Rule(self.error['NM'] & self.d_error['NS'], self.angle['NM']),  # Moving away slow → Medium negative
            ctrl.Rule(self.error['NM'] & self.d_error['ZE'], self.angle['NM']),  # Not moving → Medium negative
            ctrl.Rule(self.error['NM'] & self.d_error['PS'], self.angle['NS']),  # Moving toward slow → Small negative
            ctrl.Rule(self.error['NM'] & self.d_error['PM'], self.angle['ZE']),  # Moving toward medium → Zero
            ctrl.Rule(self.error['NM'] & self.d_error['PB'], self.angle['PS']),  # Moving toward fast → Small positive (brake)
            
            # NS (Ball slightly right of setpoint)
            ctrl.Rule(self.error['NS'] & self.d_error['NB'], self.angle['NM']),  # Moving away fast → Medium negative
            ctrl.Rule(self.error['NS'] & self.d_error['NM'], self.angle['NM']),  # Moving away medium → Medium negative
            ctrl.Rule(self.error['NS'] & self.d_error['NS'], self.angle['NS']),  # Moving away slow → Small negative
            ctrl.Rule(self.error['NS'] & self.d_error['ZE'], self.angle['NS']),  # Not moving → Small negative
            ctrl.Rule(self.error['NS'] & self.d_error['PS'], self.angle['ZE']),  # Moving toward slow → Zero
            ctrl.Rule(self.error['NS'] & self.d_error['PM'], self.angle['PS']),  # Moving toward medium → Small positive (brake)
            ctrl.Rule(self.error['NS'] & self.d_error['PB'], self.angle['PM']),  # Moving toward fast → Medium positive (brake)
            
            # ZE (Ball at setpoint)
            ctrl.Rule(self.error['ZE'] & self.d_error['NB'], self.angle['NM']),  # Moving right fast → Medium negative (brake)
            ctrl.Rule(self.error['ZE'] & self.d_error['NM'], self.angle['NS']),  # Moving right medium → Small negative
            ctrl.Rule(self.error['ZE'] & self.d_error['NS'], self.angle['NS']),  # Moving right slow → Small negative
            ctrl.Rule(self.error['ZE'] & self.d_error['ZE'], self.angle['ZE']),  # Not moving → Zero (perfect!)
            ctrl.Rule(self.error['ZE'] & self.d_error['PS'], self.angle['PS']),  # Moving left slow → Small positive
            ctrl.Rule(self.error['ZE'] & self.d_error['PM'], self.angle['PS']),  # Moving left medium → Small positive
            ctrl.Rule(self.error['ZE'] & self.d_error['PB'], self.angle['PM']),  # Moving left fast → Medium positive (brake)
            
            # PS (Ball slightly left of setpoint)
            ctrl.Rule(self.error['PS'] & self.d_error['NB'], self.angle['NM']),  # Moving right fast → Medium negative (brake)
            ctrl.Rule(self.error['PS'] & self.d_error['NM'], self.angle['NS']),  # Moving right medium → Small negative (brake)
            ctrl.Rule(self.error['PS'] & self.d_error['NS'], self.angle['ZE']),  # Moving right slow → Zero
            ctrl.Rule(self.error['PS'] & self.d_error['ZE'], self.angle['PS']),  # Not moving → Small positive
            ctrl.Rule(self.error['PS'] & self.d_error['PS'], self.angle['PS']),  # Moving left slow → Small positive
            ctrl.Rule(self.error['PS'] & self.d_error['PM'], self.angle['PM']),  # Moving left medium → Medium positive
            ctrl.Rule(self.error['PS'] & self.d_error['PB'], self.angle['PM']),  # Moving left fast → Medium positive
            
            # PM (Ball moderately left of setpoint)
            ctrl.Rule(self.error['PM'] & self.d_error['NB'], self.angle['NS']),  # Moving right fast → Small negative (brake)
            ctrl.Rule(self.error['PM'] & self.d_error['NM'], self.angle['ZE']),  # Moving right medium → Zero
            ctrl.Rule(self.error['PM'] & self.d_error['NS'], self.angle['PS']),  # Moving right slow → Small positive
            ctrl.Rule(self.error['PM'] & self.d_error['ZE'], self.angle['PM']),  # Not moving → Medium positive
            ctrl.Rule(self.error['PM'] & self.d_error['PS'], self.angle['PM']),  # Moving left slow → Medium positive
            ctrl.Rule(self.error['PM'] & self.d_error['PM'], self.angle['PB']),  # Moving left medium → Strong positive
            ctrl.Rule(self.error['PM'] & self.d_error['PB'], self.angle['PB']),  # Moving left fast → Strong positive
            
            # PB (Ball far to left of setpoint)
            ctrl.Rule(self.error['PB'] & self.d_error['NB'], self.angle['ZE']),  # Moving right fast → Zero
            ctrl.Rule(self.error['PB'] & self.d_error['NM'], self.angle['PS']),  # Moving right medium → Small positive
            ctrl.Rule(self.error['PB'] & self.d_error['NS'], self.angle['PM']),  # Moving right slow → Medium positive
            ctrl.Rule(self.error['PB'] & self.d_error['ZE'], self.angle['PB']),  # Not moving → Strong positive
            ctrl.Rule(self.error['PB'] & self.d_error['PS'], self.angle['PB']),  # Moving left slow → Strong positive
            ctrl.Rule(self.error['PB'] & self.d_error['PM'], self.angle['PB']),  # Moving left medium → Strong positive
            ctrl.Rule(self.error['PB'] & self.d_error['PB'], self.angle['PB'])   # Moving left fast → Strong positive
        ]
        self.system = ctrl.ControlSystem(rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def compute_control(self, t, r, r_dot):
        self.simulator.input['error'] = r
        self.simulator.input['delta_error'] = r_dot
        self.simulator.compute()
        return float(self.simulator.output.get('angle', 0.0))

# Test Scenarios
TEST_CASES = [
    {"name": "Standard Case",        "x0": 0.0,  "v0": 0.0,  "setpoint": 0.5},
    {"name": "Distant With Velocity", "x0": -0.8, "v0": 0.5,  "setpoint": 0.5},
    {"name": "Initial Velocity",      "x0": 0.0,  "v0": 0.3,  "setpoint": 0.5},
    {"name": "Negative Setpoint",     "x0": 0.0,  "v0": 0.0,  "setpoint": -0.3}
]

# Parameter Vector and Initial Guess
NUM_PARAMS = 15
initial_params = np.array([
    -1.0, -1.0, 0.0, 1.0, 1.0,
    -1.0, -1.0, 0.0, 1.0, 1.0,
    -0.4, -0.4, 0.0, 0.4, 0.4
])

# Objective Function
def objective(params: np.ndarray) -> float:
    ctrl = TunableFuzzyController()
    ctrl.set_membership_params(params)
    costs = []
    for case in tqdm(TEST_CASES):
        system = BallAndBeamSystem()
        t, y = system.simulate(
            initial_state=[case['x0'], case['v0']],
            control_func=lambda _, r, r_dot: ctrl.compute_control(_, r, r_dot),
            t_span=(0, 5.0)
        )
        positions = y[0]
        metrics = compute_metrics(t, positions, case['setpoint'])
        cost = 5.0 * metrics['Overshoot'] + 1.0 * metrics['Settling Time'] + 10.0 * metrics['Steady-State Error']
        costs.append(cost)
    return float(np.mean(costs))

def pso_objective(X):
    # X shape: (n_particles, n_dimensions)
    return np.array([objective(x) for x in X])

if __name__ == "__main__":
    # Evaluate initial guess
    init_cost = objective(initial_params)
    print(f"Initial cost: {init_cost:.3f}")

    # Define bounds matching universes
    lower_bounds = np.array([-1.0]*5 + [-1.0]*5 + [-0.4]*5)
    upper_bounds = np.array([ 1.0]*5 + [ 1.0]*5 + [ 0.4]*5)
    bounds = (lower_bounds, upper_bounds)

    # PSO options
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=30,
        dimensions=NUM_PARAMS,
        options=options,
        bounds=bounds,
    )

    # Run optimization
    best_cost, best_params = optimizer.optimize(pso_objective, iters=5, verbose=True, n_processes=6)
    print(f"PSO completed. Best Cost: {best_cost:.3f}")
    print(f"Best Parameters: {np.round(best_params,3)}")

