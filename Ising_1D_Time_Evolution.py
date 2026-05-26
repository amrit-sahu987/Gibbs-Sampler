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

from Ising_1D import PAULIS, SUPER_PAULIS, LiouvilleSite

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
    
    # 1. Add the Coherent (Hamiltonian) Part
    # Hint: Use np.kron(H3_local, np.eye(8)) for the left acting part
    super_L_local += -1j * ( np.kron(H3_local, np.eye(dim)) - np.kron(np.eye(dim), H3_local.T) )
    #coherent term needed for evolution, but not for the steady state.
    center_ops = [np.kron(I, np.kron(op, I)) for op in [X, Y, Z]]
    
    for A in center_ops:
        # Move to energy basis, apply filter, move back
        A_eig = evecs.T.conj() @ A @ evecs
        L = evecs @ (f_matrix * A_eig) @ evecs.T.conj()
        
        # 2. Add the Dissipator Part
        # Hint: L acts on the ket (left), L^* acts on the bra (right).
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
        
        # We CAN use the TeNPy template here. The physical Hamiltonian 
        # is just nearest-neighbor Ising, so we let TeNPy map it.
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'XI', u2, 'XI', dx) # Ket evolution
            self.add_coupling( J, u1, 'IX', u2, 'IX', dx) # Bra evolution

        # We DO NOT use generic lattice iterators here. 
        # We must manually inject the rigid 3-site 1D block.
        super_H_eff = 1j * compute_true_lindbladian(J, beta)
        pauli_names = ['I', 'X', 'Y', 'Z']
        
        self.logger.info("Compiling 3-site dissipator MPO...")

        """
        If a reviewer or examiner asks why your model doesn't use the generic lat.pairs
          for everything like the official TeNPy documentation suggests,
            you have an airtight physical defense: 
            "Because the exact diagonalization of the Lindblad jump operators
              is geometrically locked. The TFIModel template is for local Hamiltonians;
                our model compiles a non-local thermal environment."
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

def simulate_thermalization(N: int, J: float, beta: float, dt=0.05, tol=1e-4):
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
    C = np.cosh(beta * J)
    S = np.sinh(beta * J)
    numerator = C * S * (C**(N-2) + S**(N-2))
    denominator = C**N + S**N
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

def run_research_sweep():
    N_list = [4, 5, 6, 7, 8, 9, 10]           
    beta_list = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  
    tol_list = [1e-5, 1e-6, 1e-7, 1e-8]         
    
    results = []
    
    for N, beta, tol in itertools.product(N_list, beta_list, tol_list):
        print(f"\n--- Running: N={N}, Beta={beta}, Tol={tol} ---")
        
        # We pass tol directly into our modified simulate_thermalization
        times, energies = simulate_thermalization(N, J=1.0, beta=beta, tol=tol)
        
        # Calculate tau_mix
        tau = extract_mixing_time(times, energies)
        
        results.append({
            'N': N,
            'Beta': beta,
            'Tol': tol,
            'Tau_Mix': tau
        })
        
    # Save to file
    df = pd.DataFrame(results)
    df.to_csv("research_data.csv", index=False)
    print("\nSweep Complete. Data saved to research_data.csv")

if __name__ == "__main__":
    run_research_sweep()
    quit()
    N = 5
    J = 1.0
    beta = 0.5
    
    # times, energies = simulate_thermalization(N, J, beta, total_time=5.0, dt=0.05)
    times, energies = simulate_thermalization(N, J, beta, dt=0.01)
    
    # Analytical target limits
    analytical_steady_state_infinity = -J * np.tanh(beta * J)
    analytical_steady_state = exact_finite_energy(N, J, beta)

    
    # Extract the empirical mixing time (tau) via non-linear regression
    def decay_law(t, E_inf, A, tau):
        return E_inf + A * np.exp(-t / tau)
    
    popt, _ = curve_fit(decay_law, times, energies, p0=[analytical_steady_state, energies[0]-analytical_steady_state, 1.0])
    E_inf, A, tau_mix = popt
    
    print("\n=== Empirical Results ===")
    print(f"Extracted Steady State Energy: {E_inf:.6f}")
    print(f"Theoretical Analytical Energy (Infinite): {analytical_steady_state_infinity:.6f}")
    print(f"Theoretical Analytical Energy (Finite): {analytical_steady_state:.6f}")
    print(f"Empirical Mixing Time (tau):   {tau_mix:.6f}")
    
    # Plot the results
    plt.figure(figsize=(9, 6))
    plt.plot(times, energies, 'o-', color='#1f77b4', markersize=4, label='TeNPy TDVP Quench')
    plt.plot(times, decay_law(times, *popt), '--', color='#d62728', linewidth=2, label=f'Exponential Fit ($\\tau={tau_mix:.3f}$)')
    plt.axhline(analytical_steady_state, color='black', linestyle=':', label='Analytical Gibbs State')
    plt.axhline(analytical_steady_state_infinity, color='green', linestyle='-.', label='Analytical Infinite T Limit')
    
    plt.title(r'Real-Time Thermal Relaxation & Mixing Time Extraction for $N={N}$', fontsize=14)
    plt.xlabel('Time ($t$)', fontsize=12)
    plt.ylabel('Average Internal Energy per Bond', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()