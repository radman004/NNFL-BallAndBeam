import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import time


error = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'error')
d_error = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'delta_error')
angle = ctrl.Consequent(np.linspace(-0.4, 0.4, 1000), 'angle')



def print_membership(params):
    # Unpack deltas
    e_d1, e_d2, e_d3, de_d1, de_d2, de_d3, o_d1, o_d2, o_d3 = np.abs(params)

    # Reconstruct triangle points for error
    e_NL_left = -1.0
    e_NL_peak = e_NL_left + e_d1
    e_ZE_peak = e_NL_peak + e_d2
    e_PL_peak = e_ZE_peak + e_d3
    e_PL_right = 1.0

    # Reconstruct triangle points for delta_error
    de_NL_left = -1.0
    de_NL_peak = de_NL_left + de_d1
    de_ZE_peak = de_NL_peak + de_d2
    de_PL_peak = de_ZE_peak + de_d3
    de_PL_right = 1.0

    # Reconstruct triangle points for output
    o_NL_left = -0.4
    o_NL_peak = o_NL_left + o_d1
    o_ZE_peak = o_NL_peak + o_d2
    o_PL_peak = o_ZE_peak + o_d3
    o_PL_right = 0.4

    # Define universes
    x_error = np.linspace(-1, 1, 1000)
    x_d_error = np.linspace(-1, 1, 1000)
    x_angle = np.linspace(-0.4, 0.4, 1000)

    # Plot membership functions
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Error MFs
    ax1.plot(x_error, fuzz.trimf(x_error, [e_NL_left, e_NL_left, e_NL_peak]), label='NB')
    ax1.plot(x_error, fuzz.trimf(x_error, [e_NL_left, e_NL_peak, e_ZE_peak]), label='NM')
    ax1.plot(x_error, fuzz.trimf(x_error, sorted([e_NL_peak, e_ZE_peak, 0.0])), label='NS')
    ax1.plot(x_error, fuzz.trimf(x_error, [e_NL_peak, e_ZE_peak, e_PL_peak]), label='ZE')
    ax1.plot(x_error, fuzz.trimf(x_error, sorted([0.0, e_ZE_peak, e_PL_peak])), label='PS')
    ax1.plot(x_error, fuzz.trimf(x_error, sorted([e_ZE_peak, e_PL_peak, e_PL_right])), label='PM')
    ax1.plot(x_error, fuzz.trimf(x_error, sorted([e_PL_peak, e_PL_right, e_PL_right])), label='PB')
    ax1.set_title('Error Membership Functions')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Membership')
    ax1.grid(True)
    ax1.legend()

    # Delta Error MFs
    ax2.plot(x_d_error, fuzz.trimf(x_d_error, [de_NL_left, de_NL_left, de_NL_peak]), label='NB')
    ax2.plot(x_d_error, fuzz.trimf(x_d_error, [de_NL_left, de_NL_peak, de_ZE_peak]), label='NM')
    ax2.plot(x_d_error, fuzz.trimf(x_d_error, sorted([de_NL_peak, de_ZE_peak, 0.0])), label='NS')
    ax2.plot(x_d_error, fuzz.trimf(x_d_error, [de_NL_peak, de_ZE_peak, de_PL_peak]), label='ZE')
    ax2.plot(x_d_error, fuzz.trimf(x_d_error, sorted([0.0, de_ZE_peak, de_PL_peak])), label='PS')
    ax2.plot(x_d_error, fuzz.trimf(x_d_error, sorted([de_ZE_peak, de_PL_peak, de_PL_right])), label='PM')
    ax2.plot(x_d_error, fuzz.trimf(x_d_error, sorted([de_PL_peak, de_PL_right, de_PL_right])), label='PB')
    ax2.set_title('Delta Error Membership Functions')
    ax2.set_xlabel('Delta Error')
    ax2.set_ylabel('Membership')
    ax2.grid(True)
    ax2.legend()

    # Angle MFs
    ax3.plot(x_angle, fuzz.trimf(x_angle, [o_NL_left, o_NL_left, o_NL_peak]), label='NB')
    ax3.plot(x_angle, fuzz.trimf(x_angle, [o_NL_left, o_NL_peak, o_ZE_peak]), label='NM')
    ax3.plot(x_angle, fuzz.trimf(x_angle, sorted([o_NL_peak, o_ZE_peak, 0.0])), label='NS')
    ax3.plot(x_angle, fuzz.trimf(x_angle, sorted([o_NL_peak, o_ZE_peak, o_PL_peak])), label='ZE')
    ax3.plot(x_angle, fuzz.trimf(x_angle, sorted([0.0, o_ZE_peak, o_PL_peak])), label='PS')
    ax3.plot(x_angle, fuzz.trimf(x_angle, sorted([o_ZE_peak, o_PL_peak, o_PL_right])), label='PM')
    ax3.plot(x_angle, fuzz.trimf(x_angle, sorted([o_PL_peak, o_PL_right, o_PL_right])), label='PB')
    ax3.set_title('Angle Membership Functions')
    ax3.set_xlabel('Angle')
    ax3.set_ylabel('Membership')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()


    # 1. Assign membership functions to fuzzy variables
    error['NB'] = fuzz.trimf(error.universe, [e_NL_left, e_NL_left, e_NL_peak])
    error['NM'] = fuzz.trimf(error.universe, [e_NL_left, e_NL_peak, e_ZE_peak])
    error['NS'] = fuzz.trimf(error.universe, sorted([e_NL_peak, e_ZE_peak, 0.0]))
    error['ZE'] = fuzz.trimf(error.universe, [e_NL_peak, e_ZE_peak, e_PL_peak])
    error['PS'] = fuzz.trimf(error.universe, sorted([0.0, e_ZE_peak, e_PL_peak]))
    error['PM'] = fuzz.trimf(error.universe, sorted([e_ZE_peak, e_PL_peak, e_PL_right]))
    error['PB'] = fuzz.trimf(error.universe, sorted([e_PL_peak, e_PL_right, e_PL_right]))

    d_error['NB'] = fuzz.trimf(d_error.universe, [de_NL_left, de_NL_left, de_NL_peak])
    d_error['NM'] = fuzz.trimf(d_error.universe, [de_NL_left, de_NL_peak, de_ZE_peak])
    d_error['NS'] = fuzz.trimf(d_error.universe, sorted([de_NL_peak, de_ZE_peak, 0.0]))
    d_error['ZE'] = fuzz.trimf(d_error.universe, [de_NL_peak, de_ZE_peak, de_PL_peak])
    d_error['PS'] = fuzz.trimf(d_error.universe, sorted([0.0, de_ZE_peak, de_PL_peak]))
    d_error['PM'] = fuzz.trimf(d_error.universe, sorted([de_ZE_peak, de_PL_peak, de_PL_right]))
    d_error['PB'] = fuzz.trimf(d_error.universe, sorted([de_PL_peak, de_PL_right, de_PL_right]))

    angle['NB'] = fuzz.trimf(angle.universe, [o_NL_left, o_NL_left, o_NL_peak])
    angle['NM'] = fuzz.trimf(angle.universe, [o_NL_left, o_NL_peak, o_ZE_peak])
    angle['NS'] = fuzz.trimf(angle.universe, sorted([o_NL_peak, o_ZE_peak, 0.0]))
    angle['ZE'] = fuzz.trimf(angle.universe, sorted([o_NL_peak, o_ZE_peak, o_PL_peak]))
    angle['PS'] = fuzz.trimf(angle.universe, sorted([0.0, o_ZE_peak, o_PL_peak]))
    angle['PM'] = fuzz.trimf(angle.universe, sorted([o_ZE_peak, o_PL_peak, o_PL_right]))
    angle['PB'] = fuzz.trimf(angle.universe, sorted([o_PL_peak, o_PL_right, o_PL_right]))

    # 2. Now create your rules
    rules = [
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