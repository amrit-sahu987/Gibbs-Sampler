import numpy as np
import pandas as pd
from tenpy.networks.mps import MPS
from tenpy.algorithms import tdvp

from builder import DissipatorBuilder
from model import OpenQuantumModel
from observer import ThermalObserver
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

from Ising_1D import find_best_gap_fit

def run_simulation(N=5, J=1.0, g=0.0, beta=1.0, dt=0.01, tol=1e-5):
    print(f"--- Starting Simulation: N={N}, Beta={beta} ---")
    
    # Physics Engine
    builder = DissipatorBuilder(beta=beta)
    # super_L = 1j * builder.build_ising_superoperator(J) #for zero field
    super_L = 1j * builder.build_ising_superoperator(J, g) #for non-zero external field
    
    # Geometry & Model Compilation
    # Stencil defines a 1D straight line for the dissipator
    stencil = [(0,0), (1,0), (2,0)] 
    
    model_params = {
        'L': N, 'J': J, 'conserve': 'None',
        'bc_MPS': 'finite', 'bc_x': 'periodic',
        'super_L': super_L, 'stencil': stencil
    }
    model = OpenQuantumModel(model_params)
    
    # Measurement Suite
    observer = ThermalObserver(model, N)
    
    # State Initialization (Neel State)
    initial_state = [0 if i % 2 == 0 else 3 for i in range(N)]
    
    psi_t = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS, dtype=complex)
    # TDVP Engine
    tdvp_params = {
        'start_time': 0.0, 'dt': dt,
        'trunc_params': {'chi_max': 150, 'svd_min': 1e-8}
    }
    engine = tdvp.TwoSiteTDVPEngine(psi_t, model, tdvp_params)
    
    # Time Evolution Loop
    times, energies = [0.0], []
    _, e_0 = observer.measure_energy(psi_t, J)
    energies.append(e_0)
    
    step, stable_count = 0, 0

    min_sim_time = 2.5
    
    while True:
        step += 1
        engine.run()
        
        # Measure exactly using the observer
        _, e_t = observer.measure_energy(psi_t, J)
        
        times.append(engine.evolved_time)
        energies.append(e_t)
        
        if step % 10 == 0:
            print(f"Time: {engine.evolved_time:5.2f} | Energy/Bond: {e_t:.6f}")
            
        # Dynamic Termination Convergence Check
        delta_E = abs(energies[-1] - energies[-2])
        
        # Only start checking for convergence IF we have passed the minimum time gate
        if engine.evolved_time > min_sim_time and delta_E < tol:
            stable_count += 1
        else:
            # Reset if the energy suddenly drops again (system found a new decay channel)
            stable_count = 0
            
        # Require 10 consecutive flat steps to declare absolute convergence
        if stable_count >= 10:
            print(f"\nConvergence reached at t = {engine.evolved_time:.2f} (Delta E < {tol})")
            break
            
        if engine.evolved_time > 100.0:
            print("\nWarning: Maximum time limit reached.")
            break
            
    return np.array(times), np.array(energies)

# def extract_mixing_time(times, energies):
#     """
#     Fits: E(t) = E_steady + A * exp(-t / tau)
#     """
#     from scipy.optimize import curve_fit
    
#     def model_func(t, E_steady, A, tau):
#         return E_steady + A * np.exp(-t / tau)
    
#     # Initial guess
#     p0 = [energies[-1], energies[0]-energies[-1], 1.0]
#     try:
#         popt, _ = curve_fit(model_func, times, energies, p0=p0)
#         # return popt[2] # Return Tau
#         return popt[2], popt[0] # Return Tau, energy
#     except:
#         return np.nan, np.nan

def extract_mixing_time(times, energies):
    """
    Fits the TDVP relaxation data by anchoring the asymptote to the final numerical value.
    Bypasses the curve fitter entirely if the data is a flatline.
    """
    try:
        # Ignore the first 1.0 seconds to bypass the initial quench
        mask = times > 1.0 
        t_fit = times[mask]
        e_fit = energies[mask]
        
        if len(t_fit) < 3:
            return np.nan, np.nan
            
        t_fit = t_fit - t_fit[0]
        
        # ANCHOR THE ASYMPTOTE
        # The simulation converged, so the last value IS the steady state.
        e_steady_actual = e_fit[-1]
        
        # THE FLATLINE CHECK
        # If the energy hasn't changed by more than 0.001, it is a flat line.
        # It either mixed instantly (high temp) or froze entirely (low temp).
        energy_variance = np.max(e_fit) - np.min(e_fit)
        if energy_variance < 1e-3:
            print(" -> Data is a flatline. Skipping exponential fit.")
            # Return NaN for tau so it doesn't plot a garbage point, 
            # but return the actual energy so the blue squares remain accurate.
            return np.nan, e_steady_actual

        # FIT ONLY AMPLITUDE AND TAU
        # We define a new 2-parameter model where E_steady is hardcoded
        def anchored_decay(t, tau, A):
            return e_steady_actual + A * np.exp(-t / tau)
        
        A_guess = e_fit[0] - e_steady_actual
        p0 = [2.0, A_guess]
        
        # Bounds: tau must be positive, Amplitude can be anything
        bounds = ([0.001, -np.inf], [np.inf, np.inf])
        
        popt, _ = curve_fit(anchored_decay, t_fit, e_fit, p0=p0, bounds=bounds, maxfev=10000)
        
        return popt[0], e_steady_actual
        
    except Exception as e:
        print(f"Curve fit failed: {e}")
        return np.nan, np.nan

def run_g_sweep():
    # Fixed parameters
    N = 6 
    J = 1.0
    beta = 1.0
    tol = 1e-5
    dt = 0.01

    g_list = [0.0, 0.01, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    jump_keys = ['X'] 

    results = []
    builder = DissipatorBuilder(beta=beta)
    stencil = [(0,0), (1,0), (2,0)]

    for g in g_list:
        print(f"\n--- Running: g={g}, Bath={jump_keys} ---")

        # Build the restricted dissipator
        super_L = 1j * builder.build_ising_superoperator(J=J, g=g, jump_keys=jump_keys)

        # Setup Model
        model_params = {
            'L': N, 'J': J, 'conserve': 'None',
            'bc_MPS': 'finite', 'bc_x': 'periodic',
            'super_L': super_L, 'stencil': stencil
        }
        model = OpenQuantumModel(model_params)
        observer = ThermalObserver(model, N)

        # Initial Pure State (Neel)
        initial_state = [0 if i % 2 == 0 else 3 for i in range(N)]
        psi_t = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS, dtype=complex)

        # TDVP Engine
        tdvp_params = {
            'start_time': 0.0, 'dt': dt,
            'trunc_params': {'chi_max': 150, 'svd_min': 1e-8}
        }
        engine = tdvp.TwoSiteTDVPEngine(psi_t, model, tdvp_params)

        # Time Evolution Loop
        times, energies = [0.0], []
        _, e_0 = observer.measure_energy(psi_t, J)
        energies.append(e_0)

        step, stable_count = 0, 0
        while True:
            step += 1
            engine.run()
            _, e_t = observer.measure_energy(psi_t, J)

            times.append(engine.evolved_time)
            energies.append(e_t)
            
            if step % 50 == 0:
                print(f" t = {engine.evolved_time:5.2f} | E = {e_t:.6f}")

            # Dynamic convergence
            if abs(energies[-1] - energies[-2]) < tol:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= 10:
                print(f" Converged safely at t={engine.evolved_time:.2f}")
                break

            # The Dark State Timeout limit
            if engine.evolved_time >= 60.0:
                print(" Timeout reached! System trapped in Dark State.")
                break

        times = np.array(times)
        energies = np.array(energies)
        tau, energy = extract_mixing_time(times, energies)

        print(f"=> Result for g={g}: Tau_Mix = {tau:.4f}")
        results.append({'g': g, 'Tau_Mix': tau, 'End_Time': engine.evolved_time})

    # Save data
    df = pd.DataFrame(results)
    df.to_csv("x_bath_sweep.csv", index=False)
    print("\nSweep saved to x_bath_sweep.csv")
    return df

def plot_dark_state_divergence():
    """Generates the plot showing how the Hamiltonian unlocks the bath."""
    df = pd.read_csv("x_bath_sweep.csv")
    
    # Filter out the g=0 timeout if the curve_fit failed to find a valid Tau
    df_clean = df.dropna(subset=['Tau_Mix'])
    
    plt.figure(figsize=(8, 5))
    plt.plot(df_clean['g'], df_clean['Tau_Mix'], 'bo-', linewidth=2, markersize=8)
    
    # Add a dashed line pointing to infinity to represent the dark state at g=0
    if 0.0 in df['g'].values and (df[df['g']==0.0]['End_Time'].values[0] >= 50.0):
        plt.axvline(x=0.0, color='r', linestyle='--', alpha=0.6, label="Dark State Trap ($g=0$)")
        plt.annotate('Diverges to $\infty$', xy=(0.02, df_clean['Tau_Mix'].max()), 
                     xytext=(0.2, df_clean['Tau_Mix'].max()),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='red')

    plt.title("Hamiltonian-Assisted Thermalization (Pure $X$ Bath)")
    plt.xlabel("Transverse Field Strength ($g$)")
    plt.ylabel("Empirical Mixing Time $\\tau_{\\text{mix}}$")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plot4_Dark_State_Divergence.pdf", dpi=300)
    print("Saved plot to Plot4_Dark_State_Divergence.pdf")

def run_beta_sweep():
    N = 6 
    J = 1.0
    g = 0.0
    tol = 1e-3
    dt = 0.01

    # Sweeping Inverse Temperature (Cooling the bath)
    # beta_list = [0.01, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    beta_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]

    #temporary to compare with exact covariance dynamics
    # tol = 1e-3
    # beta_list = [2.0]

    
    # The physical amplitude damping bath (Emission and Absorption)
    # jump_keys = ['Sp', 'Sm'] 
    jump_keys = ['X', 'Y', 'Z'] 

    results = []
    stencil = [(0,0), (1,0), (2,0)]

    for beta in beta_list:
        # print(f"\n--- Running: Beta={beta}, Bath={jump_keys} ---")

        # builder = DissipatorBuilder(beta=beta)
        # super_L = 1j * builder.build_ising_superoperator(J=J, g=g, jump_keys=jump_keys)

        # model_params = {
        #     'L': N, 'J': J, 'conserve': 'None',
        #     'bc_MPS': 'finite', 'bc_x': 'periodic',
        #     'super_L': super_L, 'stencil': stencil
        # }
        # model = OpenQuantumModel(model_params)
        # observer = ThermalObserver(model, N)

        # # Initial Pure State (All spins up - High Energy)
        # initial_state = [0] * N
        # psi_t = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS, dtype=complex)

        # tdvp_params = {
        #     'start_time': 0.0, 'dt': dt,
        #     'trunc_params': {'chi_max': 150, 'svd_min': 1e-8}
        # }
        # engine = tdvp.TwoSiteTDVPEngine(psi_t, model, tdvp_params)

        # times, energies = [0.0], []
        # _, e_0 = observer.measure_energy(psi_t, J)
        # energies.append(e_0)

        # step, stable_count = 0, 0
        # while True:
        #     step += 1
        #     engine.run()
        #     _, e_t = observer.measure_energy(psi_t, J)

        #     times.append(engine.evolved_time)
        #     energies.append(e_t)
            
        #     if step % 50 == 0:
        #         print(f" t = {engine.evolved_time:5.2f} | E = {e_t:.6f}")

        #     if abs(energies[-1] - energies[-2]) < tol:
        #         stable_count += 1
        #     else:
        #         stable_count = 0

        #     if stable_count >= 10:
        #         print(f" Converged safely at t={engine.evolved_time:.2f}")
        #         break

        #     if engine.evolved_time >= 60.0:
        #         print(" Timeout reached! System freezing.")
        #         break

        times, energies = run_simulation(N=N, J=J, g=g, beta=beta, dt=dt, tol=tol)
        times = np.array(times)
        energies = np.array(energies)
        tau, energy = extract_mixing_time(times, energies)

        e_analytical = -J * np.tanh(beta * J)
        error = abs(energy - e_analytical)

        print(f"=> Result for Beta={beta}: Tau_Mix = {tau:.4f}")
        results.append({'Beta': beta, 'Tau_Mix': tau, 'E_Steady': energy, 'E_Analytical': e_analytical, 'Error': error})

    df = pd.DataFrame(results)
    df.to_csv("beta_sweep.csv", index=False)
    print("\nSweep saved to beta_sweep.csv")
    return df

def find_best_fit(x, y, models, independent_var_name='\\beta'):
    """
    A universal fitting function. It takes a dictionary of models containing
    their lambda functions, initial guesses (p0), and labels, and finds the best fit.
    Includes safe R^2 calculation for flat plateaus.
    """
    warnings.filterwarnings('ignore')
    
    best_r2 = -float('inf')
    best_math_label = None
    best_fit_y = None

    x_fit = np.linspace(min(x), max(x), 200)

    for name, model in models.items():
        try:
            # Check if the model dictionary provided explicit bounds; otherwise, unconstrained.
            bounds = model.get('bounds', (-np.inf, np.inf))
            
            popt, _ = curve_fit(model['func'], x, y, p0=model['p0'], bounds=bounds, maxfev=50000)
            y_pred = model['func'](x, *popt)
            
            # Robust R^2 calculation to prevent crash on flat ground-state data
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 1.0

            if r2 > best_r2:
                best_r2 = r2
                best_fit_y = model['func'](x_fit, *popt)
                best_math_label = model['label'].format(*popt)
        except Exception:
            continue 

    return x_fit, best_fit_y, best_math_label

def plot_amplitude_damping_results():
    df = pd.read_csv("beta_sweep.csv")
    df_clean = df.dropna()
    df_clean = df_clean[df_clean['Tau_Mix'] > 0] 

    # Extract numerical data
    beta_arr = df_clean['Beta'].values
    tau_arr = df_clean['Tau_Mix'].values
    e_arr = df_clean['E_Steady'].values

    J_val = 1.0  # From simulation parameters

    # --- 1. Define the Models ---
    
    # Models specifically designed for Mixing Time (always positive, diverges exponentially)
    tau_models = {
        'Quadratic Exponential': {
            'func': lambda x, a, b, c: a * np.exp(b * (x**2) + c * x),
            'p0': [1.0, 1.0, 1.0],
            'label': "Fit ($\\tau$): $\\tau \\propto {:.2f} e^{{{:.2f}\\beta^2 + {:.2f}\\beta}}$"
        },
        'Standard Exponential': {
            'func': lambda x, a, b: a * np.exp(b * x),
            'p0': [1.0, 1.0],
            'label': "Fit ($\\tau$): $\\tau \\propto {:.2f} e^{{{:.2f}\\beta}}$"
        }
    }

    # Models specifically designed for Energy (Negative, plateaus at ground state)
    e_models = {
        'Analytical Tanh': {
            'func': lambda x, a, b: a * np.tanh(b * x),
            'p0': [-1.0, 1.0],
            'bounds': ([-np.inf, 0.0], [np.inf, np.inf]), # Force b > 0 to prevent upside-down tanh
            'label': "Fit ($E$): $E \\approx {:.2f} \\tanh({:.2f}\\beta)$"
        },
        'Exponential Saturation': {
            'func': lambda x, a, b, c: a + b * np.exp(-c * x),
            'p0': [min(e_arr), max(e_arr) - min(e_arr), 1.0],
            'label': "Fit ($E$): $E \\approx {:.2f} + {:.2f} e^{{-{:.2f}\\beta}}$"
        }
    }

    # --- 2. Fit the Data using the Unified Function ---
    x_smooth, y_fit_tau, math_label_tau = find_best_fit(
        beta_arr, tau_arr, models=tau_models
    )
    
    # We use x_smooth_e just in case the energy fitter needs a different x-axis resolution, 
    # though they are identical here.
    x_smooth_e, y_fit_e, math_label_e = find_best_fit(
        beta_arr, e_arr, models=e_models
    )
    
    # Fallback if energy fit fails completely
    if y_fit_e is None:
        y_fit_e = -J_val * np.tanh(J_val * x_smooth_e)
        math_label_e = f"Fit ($E$): $E \\approx -{J_val:.2f} \\tanh({J_val:.2f}\\beta)$ (Fallback)"

    # --- 3. Exact Analytical Solution ---
    e_analytical = -J_val * np.tanh(J_val * x_smooth_e)

    # --- Plotting Setup ---
    fig, ax1 = plt.subplots(figsize=(9, 7))

    # --- Axis 1: Mixing Time ---
    color1 = 'tab:red'
    ax1.set_xlabel(r'Inverse Temperature ($\beta$)')
    ax1.set_ylabel(r'Mixing Time $\tau_{\text{mix}}$', color=color1)
    
    ax1.semilogy(beta_arr, tau_arr, 'o', color=color1, markersize=8, label=r"Data ($\tau_{\text{mix}}$)")
    if y_fit_tau is not None:
        ax1.semilogy(x_smooth, y_fit_tau, '--', color=color1, linewidth=2, label=math_label_tau)
                     
    ax1.tick_params(axis='y', labelcolor=color1)

    # --- Axis 2: Steady State Energy ---
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel(r'Asymptotic Energy ($E_{\text{steady}}$)', color=color2)
    
    ax2.plot(x_smooth_e, e_analytical, 'k-', linewidth=2.5, alpha=0.5, label=r"Exact: $E = -J \tanh(\beta J)$")
    if y_fit_e is not None:
        ax2.plot(x_smooth_e, y_fit_e, '--', color=color2, linewidth=2, label=math_label_e)
    ax2.plot(beta_arr, e_arr, 's', color=color2, markersize=7, label=r"Data ($E_{\text{steady}}$)")
                     
    ax2.tick_params(axis='y', labelcolor=color2)

    # --- Formatting & Legend Placement ---
    plt.title("Gibbs Sampler Benchmarking: TDVP vs. Analytical Solutions")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), 
               fontsize=11, ncol=2, frameon=True)

    plt.savefig("Plot_Beta_Sweep_Analytical_Comparison.pdf", dpi=300, bbox_inches='tight')
    print("\nSaved plot to Plot_Beta_Sweep_Analytical_Comparison.pdf")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_dynamics(times, energies, E_analytical, tau_mix):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot the raw simulation data
    ax.plot(times, energies, 'b-', linewidth=2, label="TDVP Simulation")
    
    # Plot the Analytical Gibbs Asymptote
    ax.axhline(y=E_analytical, color='k', linestyle='--', linewidth=2.5, 
               label=f"Analytical Asymptote ($E={E_analytical:.3f}$)")
    
    # Mark the Mixing Time
    if not np.isnan(tau_mix):
        ax.axvline(x=tau_mix, color='r', linestyle=':', linewidth=2, 
                   label=f"Mixing Time ($\\tau_{{mix}}={tau_mix:.2f}$)")
        
        # Plot a red dot exactly where tau_mix intersects the data
        idx = np.argmin(np.abs(times - tau_mix))
        ax.plot(times[idx], energies[idx], 'ro', markersize=8)

    # Formatting
    ax.set_title("Energy Relaxation Dynamics")
    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel("Internal Energy ($E$)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def extract_quarter_mixing_time(times, energies):
    """
    Extracts t_mix(1/4) by finding the exact time the normalized 
    energy distance to the steady state drops below 0.25.
    """
    try:
        # The Quench Bypass (Ignore the initial non-Markovian shockwave)
        mask = times > 1.0 
        t_fit = times[mask]
        e_fit = energies[mask]
        
        if len(t_fit) < 3:
            return np.nan, np.nan
            
        # Anchor the Steady State
        e_steady_actual = e_fit[-1]
        
        # The Flatline Check
        energy_variance = np.max(e_fit) - np.min(e_fit)
        if energy_variance < 1e-3:
            print(" -> Data is a flatline. Skipping 1/4 threshold extraction.")
            return np.nan, e_steady_actual

        # Calculate Normalized Distance d_E(t)
        # We define e_start as the highest/lowest point immediately AFTER the quench
        e_start = e_fit[0]
        max_distance = abs(e_start - e_steady_actual)
        
        distances = np.abs(e_fit - e_steady_actual) / max_distance
        
        # Find the 1/4 Threshold Crossing
        # Find all indices where the distance is less than or equal to 0.25
        indices_below_threshold = np.where(distances <= 0.25)[0]
        
        if len(indices_below_threshold) == 0:
            print("Warning: Simulation did not run long enough to reach 1/4 distance.")
            return np.nan, e_steady_actual
            
        # The 1/4 mixing time is the exact time at the first crossing index
        crossing_idx = indices_below_threshold[0]
        t_quarter = t_fit[crossing_idx]
        
        return t_quarter, e_steady_actual
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return np.nan, np.nan

if __name__ == "__main__":
    beta = 0.5
    J = 1.0
    t, e = run_simulation(N=6, J=J, g=0.0, beta=beta, dt=0.01, tol=1e-3)
    t_quarter, e_steady_actual = extract_quarter_mixing_time(t, e)
    E_analytical = -J * np.tanh(beta * J) # -J * tanh(beta * J)
    plot_simulation_dynamics(t, e, E_analytical, t_quarter)
    quit()

    # run_g_sweep()
    # plot_dark_state_divergence()

    run_beta_sweep()
    plot_amplitude_damping_results()


    


    
