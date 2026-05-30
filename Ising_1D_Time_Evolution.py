import math
from pyexpat import model
from xml.parsers.expat import model
import numpy as np
import scipy.linalg as la
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from tenpy.networks.site import Site
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS

from Ising_1D import PAULIS, SUPER_PAULIS, LiouvilleSite, solve_lindbladian

def compute_true_lindbladian(J, beta):
    I, X, Y, Z = PAULIS['I'], PAULIS['X'], PAULIS['Y'], PAULIS['Z']
    # 3-site local physical energy neighborhood
    X0, X1, X2 = np.kron(X, np.kron(I, I)), np.kron(I, np.kron(X, I)), np.kron(I, np.kron(I, X))
    H3_local = -J * (X0 @ X1 + X1 @ X2) 
    # Exact Diagonalization
    evals, evecs = la.eigh(H3_local)
    nu_matrix = evals[:, None] - evals[None, :]
    # Calculate the filter matrix exactly as before
    f_matrix = np.exp(-((beta * nu_matrix + 1)**2) / 8.0 + 1.0/8.0)
    dim = 8
    super_L_local = np.zeros((dim**2, dim**2), dtype=complex)
    # Add the Coherent (Hamiltonian) Part
    # Use np.kron(H3_local, np.eye(8)) for the left acting part
    super_L_local += -1j * ( np.kron(H3_local, np.eye(dim)) - np.kron(np.eye(dim), H3_local.T) )
    #coherent term needed for evolution, but not for the steady state.
    center_ops = [np.kron(I, np.kron(op, I)) for op in [X, Y, Z]]
    for A in center_ops:
        # Move to energy basis, apply filter, move back
        A_eig = evecs.T.conj() @ A @ evecs
        L = evecs @ (f_matrix * A_eig) @ evecs.T.conj()
        # Add the Dissipator Part
        # L acts on the ket (left), L^* acts on the bra (right).
        # What does L^\dagger L act on? 
        L_super_jump = np.kron(L, L.conj())
        L_dag_L = L.conj().T @ L
        super_L_local += (L_super_jump - 0.5 * np.kron(L_dag_L, np.eye(dim)) - 0.5 * np.kron(np.eye(dim), L_dag_L.T))
    return super_L_local

class TrueLindbladianChain(CouplingMPOModel):
    """
    1D Open Quantum System Model for TeNPy Time Evolution.
    Deviations from standard TeNPy models (like TFIModel):
    1. Enforces a strict 1D Chain because the 3-site dissipator's 
       Bohr frequencies were derived via 1D exact diagonalization.
    2. Overrides charge conservation, as the thermal bath induces 
       quantum jumps that break isolated system symmetries.
    """
    # We strictly lock the model to a 1D Chain.
    default_lattice = Chain
    force_default_lattice = True 
    def init_sites(self, model_params):
        # We read what the user passed, but if they try to conserve 
        # a standard charge like 'parity', we reject it.
        conserve = model_params.get('conserve', 'None', str)
        if conserve != 'None':
            self.logger.warning(
                f"Open systems break standard conservation laws. "
                f"Overriding conserve='{conserve}' to 'None'."
            )
        # Initialize our custom d=4 supersite
        site = LiouvilleSite()
        return site
    def init_terms(self, model_params):
        J = model_params.get('J', 1.0, 'real_or_array')
        beta = model_params.get('beta', 0.5, 'real_or_array')
        # We can use the TeNPy template here. The physical Hamiltonian 
        # is just nearest-neighbor Ising, so we let TeNPy map it.
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'XI', u2, 'XI', dx) # Ket evolution
            self.add_coupling( J, u1, 'IX', u2, 'IX', dx) # Bra evolution
        # We do not use generic lattice iterators here. 
        # We must manually inject the rigid 3-site 1D block.
        super_H_eff = 1j * compute_true_lindbladian(J, beta)
        pauli_names = ['I', 'X', 'Y', 'Z']
        self.logger.info("Compiling 3-site dissipator MPO...")
        """
        why the model doesn't use the generic lat.pairs
          for everything like the official TeNPy documentation suggests: 
            Because the exact diagonalization of the Lindblad jump operators
              is geometrically locked. The TFIModel template is for local Hamiltonians;
                our model compiles a non-local thermal environment.
        """
        for kets in itertools.product(pauli_names, repeat=3):
            K = np.kron(PAULIS[kets[0]], np.kron(PAULIS[kets[1]], PAULIS[kets[2]]))
            for bras in itertools.product(pauli_names, repeat=3):
                p1, p2, p3 = f"{kets[0]}{bras[0]}", f"{kets[1]}{bras[1]}", f"{kets[2]}{bras[2]}"
                B = np.kron(PAULIS[bras[0]], np.kron(PAULIS[bras[1]], PAULIS[bras[2]]))
                P_123 = np.kron(K, B.T)
                c = np.trace(P_123.conj().T @ super_H_eff) / 64.0
                if abs(c) > 1e-10:
                    # Explicit 1D offsets: (site 0), (site 1), (site 2)
                    self.add_multi_coupling(c, [(p1, 0, 0), (p2, 1, 0), (p3, 2, 0)])

def simulate_thermalization(N: int, J: float, beta: float, dt=0.05, tol=1e-3):
    """
    Executes the TDVP quench with dynamic termination and exact Liouville observables.
    """
    model_params = {
        'L': N, 'J': J, 'beta': beta,
        'bc_MPS': 'finite', 'bc_x': 'periodic', 'conserve': 'None'
    }
    model = TrueLindbladianChain(model_params)
    initial_state = ['0' if i % 2 == 0 else '3' for i in range(N)]
    psi_t = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS, dtype=complex)
    # The Identity vector [1, 0, 0, 1] for Tr(rho)
    id_vec = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
    psi_identity = MPS.from_product_state(model.lat.mps_sites(), [id_vec] * N, bc=model.lat.bc_MPS, dtype=complex)
    # The Pauli X vector [0, 1, 1, 0] (Vectorized form of the physical X matrix)
    x_vec = np.array([0.0, 1.0, 1.0, 0.0], dtype=complex)
    # Pre-build an MPS for each physical bond to measure Tr(X_i X_{i+1} rho)
    bond_observables = []
    for i in range(N):
        state_list = [id_vec] * N
        state_list[i] = x_vec
        state_list[(i+1)%N] = x_vec
        bond_mps = MPS.from_product_state(model.lat.mps_sites(), state_list, bc=model.lat.bc_MPS, dtype=complex)
        bond_observables.append(bond_mps)
    tdvp_params = {
        'start_time': 0.0,
        'dt': dt,
        'trunc_params': {'chi_max': 150, 'svd_min': 1e-8}
    }
    engine = tdvp.TwoSiteTDVPEngine(psi_t, model, tdvp_params)
    times = [0.0]
    energies = []
    # Calculate starting energy with the new exact method
    trace_rho_0 = psi_identity.overlap(psi_t)
    raw_e_0 = sum([-J * obs.overlap(psi_t).real for obs in bond_observables])
    energies.append((raw_e_0 / trace_rho_0.real) / N)
    print(f"\n--- Starting Real-Time Thermal Relaxation (N={N}, beta={beta}) ---")
    step = 0
    stable_count = 0
    while True:
        step += 1
        engine.run()
        # Exact Trace Normalization
        trace_rho = psi_identity.overlap(psi_t)
        # Exact Liouville Energy Measurement: << X_i X_{i+1} | rho >>
        raw_energy = sum([-J * obs.overlap(psi_t).real for obs in bond_observables])
        normalized_energy = (raw_energy / trace_rho.real) / N
        times.append(engine.evolved_time)
        energies.append(normalized_energy)
        if step % 10 == 0:
            print(f"Time: {engine.evolved_time:5.2f} | Energy/Bond: {normalized_energy:.6f}")
        # Check for convergence (Has the energy plateaued?)
        delta_E = abs(energies[-1] - energies[-2])
        if delta_E < tol:
            stable_count += 1
        else:
            stable_count = 0
        # Terminate if energy has been stable for 10 consecutive steps
        if stable_count >= 10:
            print(f"\nConvergence reached at t = {engine.evolved_time:.2f} (Delta E < {tol})")
            break
        # Hard fail-safe to prevent infinite loops
        if engine.evolved_time > 60.0:
            print("\nWarning: Reached maximum time limit without full convergence.")
            break
    return np.array(times), np.array(energies)

def exact_finite_energy(N, J, beta):
    """Calculates the exact average bond energy for a finite periodic 1D Ising ring."""
    #not the simple tanh equation because of finite size effects. We can derive this from the partition function.
    #small periodic chain means the second eigenvalue is still non-negligible 
    if N < 2:
        raise ValueError("A periodic chain must have at least 2 spins.")     
    x = math.tanh(beta * J)
    # Safely handle the topological frustration edge-case at absolute zero (T -> 0).
    # If J is negative and beta is huge, math.tanh evaluates exactly to -1.0.
    # If N is odd, this causes a 0/0 division. We resolve this using L'Hôpital's rule.
    if x == -1.0 and N % 2 != 0:
        limit_val = (2 - N) / N
        return -J * limit_val
    numerator = x + math.pow(x, N - 1)
    denominator = 1 + math.pow(x, N)
    return -J * (numerator / denominator)

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
        # We define e_start as the highest/lowest point immediately after the quench
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

def run_research_sweep():
    # N_list = [4, 5, 6, 7, 8, 9, 10]           
    N_list = [11, 12, 13, 14, 15]           
    beta_list = [0.01, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9]  
    # beta_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]  
    # tol_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]  
    N = 6
    beta = 0.5
    tol = 1e-5     
    results = []
    # for N, beta, tol in itertools.product(N_list, beta_list, tol_list):
    for M in N_list:
        print(f"\n--- Running: N={M}, Beta={beta}, Tol={tol} ---")
        times, energies = simulate_thermalization(M, J=1.0, beta=beta, tol=tol)
        # Calculate tau_mix
        # tau = extract_mixing_time(times, energies)
        t_quarter, e_steady = extract_quarter_mixing_time(times, energies)
        results.append({
            'N': M,
            'Beta': beta,
            'Tol': tol,
            # 'Tau_Mix': tau, 
            'Tau_Quarter': t_quarter,
            'Final_Energy': e_steady
        })
    for mol in tol_list:
        print(f"\n--- Running: N={N}, Beta={beta}, Tol={mol} ---")
        times, energies = simulate_thermalization(N, J=1.0, beta=beta, tol=mol)
        # Calculate tau_mix
        # tau = extract_mixing_time(times, energies)
        t_quarter, e_steady = extract_quarter_mixing_time(times, energies)
        results.append({
            'N': N,
            'Beta': beta,
            'Tol': mol,
            # 'Tau_Mix': tau, 
            'Tau_Quarter': t_quarter,
            'Final_Energy': e_steady
        })
    for m in beta_list:
        print(f"\n--- Running: N={N}, Beta={m}, Tol={tol} ---")
        times, energies = simulate_thermalization(N, J=1.0, beta=m, tol=tol)
        # Calculate tau_mix
        # tau = extract_mixing_time(times, energies)
        t_quarter, e_steady = extract_quarter_mixing_time(times, energies)
        results.append({
            'N': N,
            'Beta': m,
            'Tol': tol,
            # 'Tau_Mix': tau, 
            'Tau_Quarter': t_quarter,
            'Final_Energy': e_steady
        })
    # Save to file
    df = pd.DataFrame(results)
    df.to_csv("research_data_5.csv", index=False)
    print("\nSweep Complete. Data saved to research_data_5.csv")

def benchmark_gap_vs_mixing(N, J, beta, tol=1e-5):
    """
    Compares the theoretical relaxation time (1/Δ) from the exact Liouvillian
    against the empirical 1/4-mixing time extracted from the TDVP quench.
    """
    print(f"\n--- Benchmarking N={N}, Beta={beta} ---")
    
    # Theoretical Benchmark (Exact Diagonalization)
    e_bond, gap, e_0 = solve_lindbladian(N, J, beta)
    tau_rel = 1.0 / gap if gap > 1e-13 else np.inf
    
    # Empirical Benchmark (TDVP Time Evolution)
    times, energies = simulate_thermalization(N, J, beta, dt=0.01, tol=tol)
    tau_quarter, e_steady = extract_quarter_mixing_time(times, energies)
    
    # Data Output
    print("\n=== Thermodynamic Comparison ===")
    print(f"Liouvillian Spectral Gap (Δ):      {gap:.6e}")
    print(f"Theoretical Relaxation (1/Δ):      {tau_rel:.6f}")
    print(f"Empirical Mixing Time t_mix(1/4):  {tau_quarter:.6f}")
    
    print("\n=== Energy Comparison ===")
    print(f"dmrg energy per bond: {e_bond:.6f}")
    print(f"tdvp bond energy: {e_steady:.6f}")

    # Proportionality Check
    if not np.isnan(tau_quarter) and not np.isinf(tau_rel):
        ratio = tau_quarter / tau_rel
        print(f"Scaling Ratio (t_mix / tau_rel):   {ratio:.4f}")
    else:
        print("Scaling Ratio:                     NaN (Simulation flatlined or failed to mix)")
        
    return {
        'Beta': beta,
        'Gap': gap,
        'Tau_Rel': tau_rel,
        'Tau_Quarter': tau_quarter
    }

if __name__ == "__main__":
    benchmark_gap_vs_mixing(N=6, J=1.0, beta=0.5, tol=1e-5)
    # run_research_sweep()
    quit()
    N = 6
    J = 1.0
    beta = 2.0
    tol = 1e-5
    # times, energies = simulate_thermalization(N, J, beta, total_time=5.0, dt=0.05)
    times, energies = simulate_thermalization(N, J, beta, dt=0.01, tol=tol)
    # Analytical target limits
    # analytical_steady_state_infinity = -J * np.tanh(beta * J)
    analytical_steady_state = exact_finite_energy(N, J, beta)
    # Extract the empirical mixing time (tau) via non-linear regression
    def decay_law(t, E_inf, A, tau):
        return E_inf + A * np.exp(-t / tau)
    tau_quarter, e_steady_actual = extract_quarter_mixing_time(times, energies)
    popt, popx = curve_fit(decay_law, times, energies, p0=[analytical_steady_state, energies[0]-analytical_steady_state, 1.0])
    E_inf, A, tau_mix = popt
    print("\n=== Empirical Results ===")
    print(f"Extracted Steady State Energy: {E_inf:.6f}")
    # print(f"Theoretical Analytical Energy (Infinite): {analytical_steady_state_infinity:.6f}")
    print(f"Theoretical Analytical Energy (Finite): {analytical_steady_state:.6f}")
    print(f"Empirical Mixing Time (tau):   {tau_mix:.6f}")
    # Plot the results
    plt.figure(figsize=(9, 6))
    plt.plot(times, energies, 'o', color='#1f77b4', markersize=4, label='TeNPy TDVP Quench')
    fit_equation = f'$E(t) = {E_inf:.3f} + {A:.3f} e^{{-t / {tau_mix:.3f}}}$'
    plt.plot(times, decay_law(times, *popt), '--', color='#d62728', linewidth=2, label=f'Exponential Fit: {fit_equation}')
    plt.axhline(analytical_steady_state, color='black', linestyle=':', label='Analytical Gibbs State')
    # plt.axhline(analytical_steady_state_infinity, color='green', linestyle='-.', label='Analytical Infinite T Limit')
    if not np.isnan(tau_quarter):
        # Draw the vertical line
        plt.axvline(x=tau_quarter, color='green', linestyle='-.', linewidth=2, 
                    label=f'$t_{{mix}}(1/4) = {tau_quarter:.2f}$')
        # Find the exact Y coordinate to place a dot at the intersection
        idx = np.argmin(np.abs(times - tau_quarter))
        y_val = energies[idx]
        plt.plot(tau_quarter, y_val, 's', color='green', markersize=8)
    plt.title(f'Real-Time Thermal Relaxation & Mixing Time Extraction for N={N}', fontsize=14)
    plt.xlabel('Time ($t$)', fontsize=12)
    plt.ylabel('Average Internal Energy per Bond', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()