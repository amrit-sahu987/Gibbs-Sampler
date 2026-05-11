# """
# now go through our final code.
# act as a strict code reviewer,
# picking out every piece of bad syntax,
# bad software design and general bad practice.
# explain how, and why,
# to correct these issues,
# before making the necessary changes and refactors.
# give the updated,
# improved code with concise comments
# (only key comments that explain code relating it to the theory).
# it should be robust, extensible,
# and adhere to industry best practices.
# """

# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.linalg as la
# import itertools
# from tenpy.networks.site import Site
# from tenpy.models.lattice import Chain
# from tenpy.models.model import CouplingMPOModel
# from tenpy.algorithms import dmrg
# from tenpy.networks.mps import MPS

# # =====================================================================
# # GLOBAL DEFINITIONS: Fundamental Physics Constants
# # =====================================================================
# # Define bare Hilbert space operators strictly once
# PAULIS = {
#     'I': np.array([[1, 0], [0, 1]], dtype=complex),
#     'X': np.array([[0, 1], [1, 0]], dtype=complex),
#     'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
#     'Z': np.array([[1, 0], [0, -1]], dtype=complex)
# }

# # Pre-compute the 16 Liouville super-operators (A \rho B => A \otimes B^T)
# # These represent the d=4 "supersite" basis
# SUPER_PAULIS = {
#     f"{n_ket}{n_bra}": np.kron(m_ket, m_bra.T)
#     for n_ket, m_ket in PAULIS.items()
#     for n_bra, m_bra in PAULIS.items()
# }

# # =====================================================================
# # 1. Physics Engine: Calculating the Frustration-Free Parent Hamiltonian
# # =====================================================================
# def compute_local_parent_hamiltonian(J, beta):
#     """Calculates the positive semi-definite 64x64 Parent Hamiltonian."""
#     I, X, Y, Z = PAULIS['I'], PAULIS['X'], PAULIS['Y'], PAULIS['Z']

#     # 3-site local physical energy neighborhood
#     X0 = np.kron(X, np.kron(I, I))
#     X1 = np.kron(I, np.kron(X, I))
#     X2 = np.kron(I, np.kron(I, X))
#     H3_local = -J * (X0 @ X1 + X1 @ X2) 

#     # Exact Diagonalization (nu_matrix[m, n] = E_m - E_n)
#     evals, evecs = la.eigh(H3_local)
#     nu_matrix = evals[:, None] - evals[None, :]

#     # QDB Filter: L_{mn} = f_{mn} A_{mn}
#     f_matrix = np.exp(-((beta * nu_matrix + 1)**2) / 8.0 + 1.0/8.0)

#     # Transform matrices for the right-hand operator R = \sigma^{1/2} L^T \sigma^{-1/2}
#     sigma_half = evecs @ np.diag(np.exp(-beta * evals / 2.0)) @ evecs.T.conj()
#     sigma_inv_half = evecs @ np.diag(np.exp(beta * evals / 2.0)) @ evecs.T.conj()

#     center_ops = [np.kron(I, np.kron(op, I)) for op in [X, Y, Z]]
#     super_H_local = np.zeros((64, 64), dtype=complex)

#     for A in center_ops:
#         # 1. Physical jump operator L
#         A_eig = evecs.T.conj() @ A @ evecs
#         L_eig = f_matrix * A_eig
#         L = evecs @ L_eig @ evecs.T.conj()
        
#         # 2. The right-hand bra operator R
#         R = sigma_half @ L.T @ sigma_inv_half
        
#         # 3. Superspace Annihilator: \mathbb{L} = L \otimes I - I \otimes R
#         L_super = np.kron(L, np.eye(8))
#         R_super = np.kron(np.eye(8), R)
#         LL = L_super - R_super
        
#         # 4. Add \frac{1}{2} \mathbb{L}^\dagger \mathbb{L} to the Parent Hamiltonian
#         super_H_local += 0.5 * (LL.conj().T @ LL)

#     return super_H_local

# # =====================================================================
# # 2. TeNPy Architecture: The Custom Liouville Site
# # =====================================================================
# class LiouvilleSite(Site): 
#     """A custom d=4 TeNPy site representing the vectorized ket-bra state."""
#     def __init__(self):
#         import tenpy.linalg.charges as charges
#         # Open systems exchange energy; particle/charge conservation is broken.
#         leg = charges.LegCharge.from_trivial(4)
#         super().__init__(leg, ['0', '1', '2', '3'], **SUPER_PAULIS)

# # =====================================================================
# # 3. TeNPy Architecture: The Parent Hamiltonian Model
# # =====================================================================
# class LindbladianParentModel(CouplingMPOModel):
#     """1D lattice model compiling 3-site dissipative interactions into an MPO."""
    
#     def init_lattice(self, model_params):
#         self.N = model_params.get('N', 12)
#         self.J = model_params.get('J', 1.0)
#         self.beta = model_params.get('beta', 0.5)
#         self.bc = model_params.get('bc', 'periodic') # Dynamically handle boundary
        
#         site = LiouvilleSite()
#         return Chain(self.N, site, bc=self.bc)

#     def init_terms(self, model_params):
#         """Projects the physics and adds the couplings to the MPO."""
#         print("Calculating exact Parent Hamiltonian...")
#         super_H = compute_local_parent_hamiltonian(self.J, self.beta)
#         super_H -= np.eye(64) * 1.0 # for spectral gap
        
#         print("Projecting into Pauli basis and building TeNPy MPO...")
#         pauli_names = ['I', 'X', 'Y', 'Z']
        
#         # Iterate over all 64 possible ket strings and 64 possible bra strings
#         for kets in itertools.product(pauli_names, repeat=3):
#             for bras in itertools.product(pauli_names, repeat=3):
                
#                 # Reconstruct the TeNPy super-operator names for the lattice
#                 p1 = f"{kets[0]}{bras[0]}"
#                 p2 = f"{kets[1]}{bras[1]}"
#                 p3 = f"{kets[2]}{bras[2]}"
                
#                 # Build the 8x8 physical ket operator for the 3 sites
#                 K = np.kron(PAULIS[kets[0]], np.kron(PAULIS[kets[1]], PAULIS[kets[2]]))
                
#                 # Build the 8x8 physical bra operator for the 3 sites
#                 B = np.kron(PAULIS[bras[0]], np.kron(PAULIS[bras[1]], PAULIS[bras[2]]))
                
#                 # Combine them into the exact (All-Kets \otimes All-Bras) superspace basis
#                 # Vectorization rule: A \rho B => A \otimes B^T
#                 P_123 = np.kron(K, B.T)
                
#                 # Trace inner product calculates the analytical coefficient flawlessly
#                 c = np.trace(P_123.conj().T @ super_H) / 64.0
                
#                 if abs(c) > 1e-10:
#                     # Parent Hamiltonian is strictly positive (E >= 0), add +c
#                     self.add_multi_coupling(c.real, [(p1, 0, 0), (p2, 1, 0), (p3, 2, 0)])
# # =====================================================================
# # 4. Simulation Execution
# # =====================================================================
# def run_simulation():
#     # Single source of truth for the simulation physics and geometry
#     model_params = {
#         'N': 16, 
#         'J': 1.0, 
#         'beta': 0.8,
#         'bc': 'periodic' # Toggle to 'open' and the whole script dynamically adapts
#     }
#     model = LindbladianParentModel(model_params)
    
#     # DMRG Entanglement parameters adapt to boundary condition constraints
#     chi_max = 250 if model_params['bc'] == 'periodic' else 100
#     dmrg_params = {
#         'mixer': True, 
#         'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-8},
#         'max_sweeps': 12,
#     }
    
#     # Initialize unentangled product state to prevent tensor conditioning errors
#     initial_state = ['0'] * model.lat.N_sites
#     psi = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS)
    
#     print(f"\nStarting TeNPy DMRG | N={model_params['N']} | beta={model_params['beta']} | BC={model_params['bc'].upper()}")
#     info = dmrg.run(psi, model, dmrg_params)
    
#     # --- Observable Extraction ---
#     # In Liouville space, Tr(rho * Op) equates to << I | Op_ket | rho >>
#     # 'XI' means apply Pauli X to the physical ket, and Identity to the bra.
#     total_energy = 0.0
#     N = model_params['N']
#     bonds_to_measure = N if model_params['bc'] == 'periodic' else N - 1
    
#     for i in range(bonds_to_measure):
#         # Calculate neighbor index with safe modulo wrapping for PBC
#         j = (i + 1) % N
#         bond_energy = psi.expectation_value_term([('XI', i), ('XI', j)])
#         total_energy += -model_params['J'] * bond_energy.real
        
#     print("\n--- Simulation Results ---")
#     print(f"Converged Steady-State Energy: {total_energy:.4f}")
#     print(f"Average Energy per Bond:       {total_energy / bonds_to_measure:.4f}")

# # (Assume PAULIS, SUPER_PAULIS, LiouvilleSite, and LindbladianParentModel 
# # are defined exactly as they were in the previous refactored code block)

# def calculate_gap_scaling(N_list, J=1.0, beta=0.8):
#     gaps = []
    
#     print(f"--- Starting Spectral Gap Scaling Analysis ---")
#     print(f"Parameters: J={J}, beta={beta}\n")
    
#     for N in N_list:
#         print(f"========== System Size N = {N} ==========")
#         model_params = {'N': N, 'J': J, 'beta': beta, 'bc': 'periodic'}
#         model = LindbladianParentModel(model_params)
        
#         # Base DMRG Parameters
#         dmrg_params = {
#             'mixer': True, 
#             'trunc_params': {'chi_max': 200, 'svd_min': 1e-8},
#             'max_sweeps': 10,
#             'verbose': 0 # Turn off step-by-step logs to keep console clean
#         }
        
#         # ---------------------------------------------------------
#         # Step 1: Find the Steady State (Ground State of -H_parent)
#         # ---------------------------------------------------------
#         initial_state = ['0'] * N
#         psi_0 = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS)
        
#         print("Optimizing Ground State (Steady State)...")
#         info_0 = dmrg.run(psi_0, model, dmrg_params)
#         E_0 = info_0['E']
        
#         # VERIFICATION 1: Is the eigenvalue exactly the shifted vacuum (-N)?
#         expected_E0 = -1.0 * N
#         assert abs(E_0 - expected_E0) < 1e-5, f"WARNING: E_0 is not {expected_E0}! Found E_0 = {E_0}"
        
#         # VERIFICATION 2: Does it match 1D Ising Thermodynamics?
#         total_energy = 0.0
#         for i in range(N):
#             bond_energy = psi_0.expectation_value_term([('XI', i), ('XI', (i + 1) % N)])
#             total_energy += -J * bond_energy.real
            
#         mps_energy_per_bond = total_energy / N
#         analytical_energy = -J * np.tanh(beta * J)
        
#         print(f" [Verify] Ground State Energy (E_0): {E_0:.4f} (Shifted to -N)")
#         print(f" [Verify] MPS Physical Energy/Bond:  {mps_energy_per_bond:.6f}")
#         print(f" [Verify] Analytical Ising Energy:   {analytical_energy:.6f}")
        
#         # ---------------------------------------------------------
#         # Step 2: Find the First Excited State (Slowest decay mode)
#         # ---------------------------------------------------------
#         initial_state_1 = ['1'] * N
#         psi_1 = MPS.from_product_state(model.lat.mps_sites(), initial_state_1, bc=model.lat.bc_MPS)
        
#         # Clean up the dictionary so TeNPy stops throwing the warning
#         if 'orthogonal_to' in dmrg_params:
#             del dmrg_params['orthogonal_to']
#         if 'verbose' in dmrg_params:
#             del dmrg_params['verbose']
        
#         print("Optimizing First Excited State...")
        
#         # THE FIX: Pass orthogonal_to as a direct keyword argument. 
#         # By passing a list of MPS objects, TeNPy automatically applies the energy penalty!
#         info_1 = dmrg.run(psi_1, model, dmrg_params, orthogonal_to=[psi_0])
#         E_1 = info_1['E']
        
#         # ---------------------------------------------------------
#         # Step 3: Compute the Gap
#         # ---------------------------------------------------------
#         gap = E_1 - E_0
#         gaps.append(gap)
#         print(f" => Spectral Gap for N={N}: {gap:.6f}\n")

#     return N_list, gaps

# # =====================================================================
# # Execution and Plotting
# # =====================================================================
# if __name__ == "__main__":
#     # run_simulation()
#     # Sweep over even system sizes
#     N_values = [4, 6, 8, 10, 12]
    
#     N_list, gap_list = calculate_gap_scaling(N_values, J=1.0, beta=0.5)
    
#     # Plotting the results
#     plt.figure(figsize=(8, 5))
    
#     # We often plot gaps on a log scale to easily spot exponential/algebraic closure
#     plt.plot(N_list, gap_list, marker='o', linestyle='-', color='b', linewidth=2)
#     plt.yscale('log') 
    
#     plt.title(r'Liouvillian Spectral Gap ($\Delta$) vs. System Size ($N$)', fontsize=14)
#     plt.xlabel(r'System Size $N$', fontsize=12)
#     plt.ylabel(r'Spectral Gap $\Delta = E_1 - E_0$', fontsize=12)
#     plt.grid(True, which="both", ls="--", alpha=0.5)
    
#     plt.tight_layout()
#     plt.show()
    
import itertools
import logging
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from scipy.optimize import curve_fit

from tenpy.networks.site import Site
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

# Setup basic logging to suppress unnecessary warnings while keeping fatal errors
logging.basicConfig(level=logging.INFO, format='%(message)s')

# =====================================================================
# GLOBAL PHYSICS CONSTANTS
# =====================================================================
PAULIS = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex)
}

SUPER_PAULIS = {
    f"{n_ket}{n_bra}": np.kron(m_ket, m_bra.T)
    for n_ket, m_ket in PAULIS.items()
    for n_bra, m_bra in PAULIS.items()
}

LANCZOS_ENERGY_SHIFT = 1.0  # Global shift to bypass TeNPy E=0 early termination

# =====================================================================
# 1. Physics Engine
# =====================================================================
def compute_local_parent_hamiltonian(J: float, beta: float) -> np.ndarray:
    """Calculates the positive semi-definite Frustration-Free Parent Hamiltonian."""
    I, X, Y, Z = PAULIS['I'], PAULIS['X'], PAULIS['Y'], PAULIS['Z']

    # 3-site local physical energy neighborhood
    X0, X1, X2 = np.kron(X, np.kron(I, I)), np.kron(I, np.kron(X, I)), np.kron(I, np.kron(I, X))
    H3_local = -J * (X0 @ X1 + X1 @ X2) 

    # Exact Diagonalization
    evals, evecs = la.eigh(H3_local)
    nu_matrix = evals[:, None] - evals[None, :]

    # QDB Filter
    f_matrix = np.exp(-((beta * nu_matrix + 1)**2) / 8.0 + 1.0/8.0)

    # Thermal Similarity Transformations
    sigma_half = evecs @ np.diag(np.exp(-beta * evals / 2.0)) @ evecs.T.conj()
    sigma_inv_half = evecs @ np.diag(np.exp(beta * evals / 2.0)) @ evecs.T.conj()

    center_ops = [np.kron(I, np.kron(op, I)) for op in [X, Y, Z]]
    dim = 8
    super_H_local = np.zeros((dim**2, dim**2), dtype=complex)

    for A in center_ops:
        A_eig = evecs.T.conj() @ A @ evecs
        L = evecs @ (f_matrix * A_eig) @ evecs.T.conj()
        R = sigma_half @ L.T @ sigma_inv_half
        
        # Frustration-free annihilator
        LL = np.kron(L, np.eye(dim)) - np.kron(np.eye(dim), R)
        super_H_local += 0.5 * (LL.conj().T @ LL)

    return super_H_local

# =====================================================================
# 2. TeNPy Architecture
# =====================================================================
class LiouvilleSite(Site): 
    """A custom d=4 TeNPy site representing the vectorized ket-bra state."""
    def __init__(self):
        import tenpy.linalg.charges as charges
        leg = charges.LegCharge.from_trivial(4)
        super().__init__(leg, ['0', '1', '2', '3'], **SUPER_PAULIS)


class LindbladianParentModel(CouplingMPOModel):
    """1D lattice model compiling 3-site dissipative interactions."""
    def init_lattice(self, model_params: Dict):
        self.N = model_params.get('N', 12)
        self.J = model_params.get('J', 1.0)
        self.beta = model_params.get('beta', 0.5)
        self.bc = model_params.get('bc', 'periodic')
        return Chain(self.N, LiouvilleSite(), bc=self.bc)

    def init_terms(self, model_params: Dict):
        super_H = compute_local_parent_hamiltonian(self.J, self.beta)
        
        # Apply strict energy shift to bypass the Lanczos zero-energy trap
        super_H -= np.eye(super_H.shape[0]) * LANCZOS_ENERGY_SHIFT
        
        pauli_names = ['I', 'X', 'Y', 'Z']
        
        for kets in itertools.product(pauli_names, repeat=3):
            K = np.kron(PAULIS[kets[0]], np.kron(PAULIS[kets[1]], PAULIS[kets[2]]))
            for bras in itertools.product(pauli_names, repeat=3):
                p1, p2, p3 = f"{kets[0]}{bras[0]}", f"{kets[1]}{bras[1]}", f"{kets[2]}{bras[2]}"
                
                B = np.kron(PAULIS[bras[0]], np.kron(PAULIS[bras[1]], PAULIS[bras[2]]))
                P_123 = np.kron(K, B.T)
                
                c = np.trace(P_123.conj().T @ super_H) / 64.0
                if abs(c) > 1e-10:
                    self.add_multi_coupling(c.real, [(p1, 0, 0), (p2, 1, 0), (p3, 2, 0)])

# =====================================================================
# 3. Core Simulation API
# =====================================================================
def solve_lindbladian(N: int, J: float, beta: float) -> Tuple[float, float, float]:
    """Runs DMRG to find the steady state and spectral gap of the system."""
    model_params = {'N': N, 'J': J, 'beta': beta, 'bc': 'periodic'}
    model = LindbladianParentModel(model_params)
    
    dmrg_config = {
        'mixer': True, 
        'trunc_params': {'chi_max': 200, 'svd_min': 1e-8},
        'max_sweeps': 10,
        'active_sites': 2
    }
    
    # ---------------------------------------------------------
    # 1. Compute Steady State (Ground State)
    # ---------------------------------------------------------
    psi_0 = MPS.from_product_state(model.lat.mps_sites(), ['0'] * N, bc=model.lat.bc_MPS)
    
    # FIX: Pass dmrg_config strictly as the 3rd positional argument
    info_0 = dmrg.run(psi_0, model, dmrg_config)
    E_0 = info_0['E']
    
    # Mathematical Validation
    expected_E0 = -LANCZOS_ENERGY_SHIFT * N
    assert abs(E_0 - expected_E0) < 1e-4, f"Integrity Failure: E_0 is {E_0}, expected {expected_E0}"
    
    # Physical Observable: Energy per bond
    total_energy = sum([-J * psi_0.expectation_value_term([('XI', i), ('XI', (i+1)%N)]).real for i in range(N)])
    energy_per_bond = total_energy / N
    
    # ---------------------------------------------------------
    # 2. Compute Slowest Decay Mode (First Excited State)
    # ---------------------------------------------------------
    psi_1 = MPS.from_product_state(model.lat.mps_sites(), ['1'] * N, bc=model.lat.bc_MPS)
    
    # FIX: Pass dmrg_config as the 3rd positional argument. 
    # orthogonal_to remains a clean keyword argument!
    info_1 = dmrg.run(psi_1, model, dmrg_config, orthogonal_to=[psi_0])
    E_1 = info_1['E']
    
    gap = E_1 - E_0
    return energy_per_bond, gap, E_0

def thermodynamic_extrapolation(N_list, gap_list):
    """
    Performs Finite-Size Scaling to find the thermodynamic limit of the gap.
    """
    N_arr = np.array(N_list)
    gap_arr = np.array(gap_list)
    inv_N = 1.0 / N_arr

    # 1. Define our theoretical scaling models
    def scaling_law(N, Delta_inf, A, exponent):
        """ Fits Delta(N) = Delta_inf + A * (1/N)^exponent """
        return Delta_inf + A * (1.0 / N)**exponent

    # 2. Perform the non-linear curve fit
    # We provide initial guesses: Delta_inf=0.1, A=1.0, exponent=1.0
    popt, pcov = curve_fit(scaling_law, N_arr, gap_arr, p0=[0.1, 1.0, 1.0], maxfev=10000)
    Delta_inf, A, exponent = popt
    
    print("\n=== Finite-Size Scaling Results ===")
    print(f"Extrapolated Gap Limit (N -> inf) : {Delta_inf:.6f}")
    print(f"Finite-Size Correction Amplitude  : {A:.6f}")
    print(f"Scaling Exponent                  : {exponent:.6f}")

    # 3. Generate high-resolution data for the fit line
    N_fit = np.linspace(min(N_arr), 1000, 500) # Go up to N=1000 to simulate infinity
    inv_N_fit = 1.0 / N_fit
    gap_fit = scaling_law(N_fit, *popt)

    # 4. Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot the simulated data points
    plt.scatter(inv_N, gap_arr, color='red', s=100, zorder=5, label='TeNPy DMRG Data')
    
    # Plot the extrapolated fit line
    plt.plot(inv_N_fit, gap_fit, color='blue', linestyle='--', label=f'Fit: $\Delta_\infty$ = {Delta_inf:.4f}')
    
    # Highlight the Y-intercept (The Thermodynamic Limit)
    plt.scatter([0], [Delta_inf], color='gold', edgecolor='black', s=150, zorder=6, label='Thermodynamic Limit ($1/N = 0$)')

    plt.title('Finite-Size Scaling of the Liouvillian Spectral Gap', fontsize=14)
    plt.xlabel('Inverse System Size ($1/N$)', fontsize=12)
    plt.ylabel('Spectral Gap $\Delta$', fontsize=12)
    plt.xlim(-0.05, max(inv_N) + 0.05) # Show slightly past the y-axis
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# Example usage to drop at the bottom of your main block:
# N_list = [4, 6, 8, 10, 12]
# gap_list = [your_computed_gaps]
# thermodynamic_extrapolation(N_list, gap_list)
# =====================================================================
# 4. Execution & Visualization
# =====================================================================
if __name__ == "__main__":
    N_values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    J_val, beta_val = 1.0, 0.5
    gap_list = []
    
    logging.info(f"--- Running Spectral Gap Scaling (J={J_val}, beta={beta_val}) ---")
    
    for N in N_values:
        logging.info(f"\n[System Size N={N}]")
        e_bond, gap, e_0 = solve_lindbladian(N, J_val, beta_val)
        gap_list.append(gap)
        
        logging.info(f"  Steady State Energy E_0 : {e_0:.4f}")
        logging.info(f"  Physical Energy / Bond  : {e_bond:.6f} (Limit: {-J_val * np.tanh(beta_val * J_val):.6f})")
        logging.info(f"  Spectral Gap (Delta)    : {gap:.6f}")

    thermodynamic_extrapolation(N_values, gap_list)
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, gap_list, marker='o', linestyle='-', color='#d32f2f', linewidth=2)
    plt.yscale('log') 
    plt.title(r'Liouvillian Spectral Gap ($\Delta$) vs. System Size ($N$)', fontsize=14)
    plt.xlabel(r'System Size $N$', fontsize=12)
    plt.ylabel(r'Spectral Gap $\Delta$', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()