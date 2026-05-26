import numpy as np
import pandas as pd
from tenpy.networks.mps import MPS
from tenpy.algorithms import tdvp

from builder import DissipatorBuilder
from model import OpenQuantumModel
from observer import ThermalObserver
import matplotlib.pyplot as plt

def run_simulation(N=5, J=1.0, g=0.5, beta=1.0, dt=0.01, tol=1e-5):
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
        if abs(energies[-1] - energies[-2]) < tol:
            stable_count += 1
        else:
            stable_count = 0
            
        if stable_count >= 10:
            print(f"\nConvergence reached at t = {engine.evolved_time:.2f} (Delta E < {tol})")
            break
            
        if engine.evolved_time > 100.0:
            print("\nWarning: Maximum time limit reached.")
            break
            
    return np.array(times), np.array(energies)

def extract_mixing_time(times, energies):
    """
    Fits: E(t) = E_steady + A * exp(-t / tau)
    """
    from scipy.optimize import curve_fit
    
    def model_func(t, E_steady, A, tau):
        return E_steady + A * np.exp(-t / tau)
    
    # Initial guess
    p0 = [energies[-1], energies[0]-energies[-1], 1.0]
    try:
        popt, _ = curve_fit(model_func, times, energies, p0=p0)
        return popt[2] # Return Tau
    except:
        return np.nan

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
        tau = extract_mixing_time(times, energies)

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
    g = 0.5
    tol = 1e-6
    dt = 0.01

    # Sweeping Inverse Temperature (Cooling the bath)
    beta_list = [0.1, 0.5, 1.0, 2.0, 3.0]
    
    # The physical amplitude damping bath (Emission and Absorption)
    jump_keys = ['Sp', 'Sm'] 

    results = []
    stencil = [(0,0), (1,0), (2,0)]

    for beta in beta_list:
        print(f"\n--- Running: Beta={beta}, Bath={jump_keys} ---")

        builder = DissipatorBuilder(beta=beta)
        super_L = 1j * builder.build_ising_superoperator(J=J, g=g, jump_keys=jump_keys)

        model_params = {
            'L': N, 'J': J, 'conserve': 'None',
            'bc_MPS': 'finite', 'bc_x': 'periodic',
            'super_L': super_L, 'stencil': stencil
        }
        model = OpenQuantumModel(model_params)
        observer = ThermalObserver(model, N)

        # Initial Pure State (All spins up - High Energy)
        initial_state = [0] * N
        psi_t = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS, dtype=complex)

        tdvp_params = {
            'start_time': 0.0, 'dt': dt,
            'trunc_params': {'chi_max': 150, 'svd_min': 1e-8}
        }
        engine = tdvp.TwoSiteTDVPEngine(psi_t, model, tdvp_params)

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

            if abs(energies[-1] - energies[-2]) < tol:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= 10:
                print(f" Converged safely at t={engine.evolved_time:.2f}")
                break

            if engine.evolved_time >= 60.0:
                print(" Timeout reached! System freezing.")
                break

        times = np.array(times)
        energies = np.array(energies)
        tau = extract_mixing_time(times, energies)

        print(f"=> Result for Beta={beta}: Tau_Mix = {tau:.4f}")
        results.append({'Beta': beta, 'Tau_Mix': tau})

    df = pd.DataFrame(results)
    df.to_csv("amplitude_damping_sweep.csv", index=False)
    print("\nSweep saved to amplitude_damping_sweep.csv")
    return df

def plot_amplitude_damping_results():
    df = pd.read_csv("amplitude_damping_sweep.csv")
    df_clean = df.dropna()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Mixing Time (Diverging as bath freezes)
    color = 'tab:red'
    ax1.set_xlabel('Inverse Temperature ($\\beta$)')
    ax1.set_ylabel('Mixing Time $\\tau_{\\text{mix}}$', color=color)
    ax1.plot(df_clean['Beta'], df_clean['Tau_Mix'], 'o-', color=color, linewidth=2, label="Mixing Time")
    ax1.tick_params(axis='y', labelcolor=color)

    # Steady State Energy (Approaching Ground State)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Asymptotic Energy ($E_{\\text{steady}}$)', color=color)
    ax2.plot(df_clean['Beta'], df_clean['E_Steady'], 's--', color=color, linewidth=2, label="Steady State Energy")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Amplitude Damping: Cooling to the Ground State")
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("Plot5_Amplitude_Damping.pdf", dpi=300)
    print("Saved plot to Plot5_Amplitude_Damping.pdf")

if __name__ == "__main__":
    # t, e = run_simulation()

    # run_g_sweep()
    # plot_dark_state_divergence()

    run_beta_sweep()
    plot_amplitude_damping_results()


    


    
