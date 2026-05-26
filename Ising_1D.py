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

logging.basicConfig(level=logging.INFO, format='%(message)s')

# GLOBAL PHYSICS CONSTANTS
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

# Physics Engine
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

# TeNPy Architecture
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

# Core Simulation API
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
    
    # Compute Steady State (Ground State)

    # psi_0 = MPS.from_product_state(model.lat.mps_sites(), ['0'] * N, bc=model.lat.bc_MPS)
    initial_state_0 = np.random.choice(['0', '1', '2', '3'], size=N).tolist()
    psi_0 = MPS.from_product_state(model.lat.mps_sites(), initial_state_0, bc=model.lat.bc_MPS)

    info_0 = dmrg.run(psi_0, model, dmrg_config)
    E_0 = info_0['E']
    
    # Mathematical Validation
    expected_E0 = -LANCZOS_ENERGY_SHIFT * N
    assert abs(E_0 - expected_E0) < 1e-4, f"Integrity Failure: E_0 is {E_0}, expected {expected_E0}"
    
    # Physical Observable: Energy per bond
    total_energy = sum([-J * psi_0.expectation_value_term([('XI', i), ('XI', (i+1)%N)]).real for i in range(N)])
    energy_per_bond = total_energy / N
    
    # Compute Slowest Decay Mode (First Excited State)
    # psi_1 = MPS.from_product_state(model.lat.mps_sites(), ['1'] * N, bc=model.lat.bc_MPS)
    initial_state_1 = np.random.choice(['0', '1', '2', '3'], size=N).tolist()
    psi_1 = MPS.from_product_state(model.lat.mps_sites(), initial_state_1, bc=model.lat.bc_MPS)
    
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

    # Define our theoretical scaling models
    def scaling_law(N, Delta_inf, A, exponent):
        """ Fits Delta(N) = Delta_inf + A * (1/N)^exponent """
        return Delta_inf + A * (1.0 / N)**exponent

    # Perform the non-linear curve fit
    # We provide initial guesses: Delta_inf=0.1, A=1.0, exponent=1.0
    popt, pcov = curve_fit(scaling_law, N_arr, gap_arr, p0=[0.1, 1.0, 1.0], maxfev=10000)
    Delta_inf, A, exponent = popt
    
    print("\n=== Finite-Size Scaling Results ===")
    print(f"Extrapolated Gap Limit (N -> inf) : {Delta_inf:.6f}")
    print(f"Finite-Size Correction Amplitude  : {A:.6f}")
    print(f"Scaling Exponent                  : {exponent:.6f}")

    # Generate high-resolution data for the fit line
    N_fit = np.linspace(min(N_arr), 1000, 500) # Go up to N=1000 to simulate infinity
    inv_N_fit = 1.0 / N_fit
    gap_fit = scaling_law(N_fit, *popt)

    # Plotting
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
    # plt.xlim(-0.05, max(inv_N) + 0.05) # Show slightly past the y-axis
    plt.xlim(0, max(inv_N) + 0.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# 4. Execution & Visualization
if __name__ == "__main__":
    N_values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    N_values = [5]
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

    '''
    start with drho/dt = L[rho]
    use forward euler with small time step
    solution is e^Lt
    can do a bigger time step with more eulers, taylor expand the exponential
    find a good cutoff with more orders vs bigger timestep
    find energy at each time step by hamiltonian of system on rho (trace)
    look at energy over time, exponentially decreasing towards gibbs state
    spectral decomposition of L decays exponentially, fit exponential to tail of the convergence
    10^-3 hartree (energy system)
    relative error of 0.1 or 0.01 should be good enough
    save the state as an MPO 

    test convergence by reducing the time step
    look at trace of density matrix, should be 1 (normalised rho) and trace(Lrho) should be 0 (steady state)
    if you run out of memory, reduce the bond dimension by compressing
    '''