import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import time


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


# Plot membership functions
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

error.view(ax=axs[0])
axs[0].set_title('Error Membership Functions')

d_error.view(ax=axs[1])
axs[1].set_title('Delta Error Membership Functions')

angle.view(ax=axs[2])
axs[2].set_title('Angle Membership Functions')

plt.tight_layout()
plt.show()