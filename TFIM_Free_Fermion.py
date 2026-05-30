import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def build_h_matrix(N: int, J: float, g: float) -> np.ndarray:
    """
    Constructs the 2N x 2N single-particle Hamiltonian matrix h.
    Enforces Open Boundary Conditions (OBC) and the strict 1/2 factor 
    to prevent double-counting in the Majorana basis.
    """
    dim = 2 * N
    h = np.zeros((dim, dim), dtype=complex)
    
    # 1. Transverse Field (g)
    for j in range(N):
        idx1 = 2 * j
        idx2 = 2 * j + 1
        h[idx1, idx2] = -0.5j * g
        h[idx2, idx1] =  0.5j * g
        
    # 2. Ising Interaction (J) - Open Boundary Conditions
    for j in range(N - 1):
        idx1 = 2 * j + 1
        idx2 = 2 * j + 2
        h[idx1, idx2] = -0.5j * J
        h[idx2, idx1] =  0.5j * J
        
    return h

def compute_evolution_matrices(h: np.ndarray, beta: float):
    """
    Diagonalizes h to apply the Quantum Detailed Balance filter,
    then analytically computes the M and N drift/drive matrices.
    """
    # 1. Exact Diagonalization
    # h is strictly Hermitian and purely imaginary, yielding real eigenvalues
    evals, V = la.eigh(h)
    
    # 2. Apply the Filter Function to the Bohr Frequencies (-4 * evals)
    # f(v) = exp( - (beta*v + 1)^2 / 8 + 1/8 )
    v = -4.0 * evals
    filter_eigenvalues = np.exp(-((beta * v + 1.0)**2) / 8.0 + 1.0 / 8.0)
    
    # 3. Reconstruct the Filter Matrix (F) and Dissipator Matrix (M)
    # F = V * diag(f) * V^dagger
    F = V @ np.diag(filter_eigenvalues) @ V.conj().T
    
    # M = F^2 (Since F is Hermitian, F^dagger F = F^2)
    M = F @ F
    
    # 4. Construct the Covariance Evolution Matrices
    # M_mat = -4i*h - 2*Re(M)
    M_mat = -4.0j * h - 2.0 * np.real(M)
    
    # N_mat = 4*M
    N_mat = 4.0 * M
    
    return M_mat, N_mat

def solve_covariance_dynamics(N: int, J: float, g: float, beta: float, t_max: float):
    """
    Integrates the dC/dt differential equation to track real-time thermalization.
    """
    h = build_h_matrix(N, J, g)
    M_mat, N_mat = compute_evolution_matrices(h, beta)
    
    # Initial State: Infinite Temperature (Completely Mixed)
    # C_ab = Tr(rho_inf * w_a w_b) = delta_ab
    C0 = np.eye(2 * N, dtype=complex)
    
    def dC_dt(t, C_flat):
        C = C_flat.reshape((2 * N, 2 * N))
        # The exact Covariance Matrix Adjoint Master Equation
        dC = M_mat @ C + C @ M_mat.T + N_mat
        return dC.flatten()

    print(f"Simulating N={N} spins (beta={beta}) up to t={t_max}...")
    
    # Solve the ODE
    sol = solve_ivp(dC_dt, [0, t_max], C0.flatten(), 
                    t_eval=np.linspace(0, t_max, 300), 
                    method='RK45', rtol=1e-7, atol=1e-9)
    
    times = sol.t
    energies = []
    
    # Extract the physical internal energy at each time step: E(t) = Tr(h * C(t))
    for C_flat in sol.y.T:
        C = C_flat.reshape((2 * N, 2 * N))
        # E_t = np.trace(h @ C)
        # Compute exact energy using element-wise multiplication
        E_t = np.sum(h * C)
        # Energy must be purely real; we extract the real part
        energies.append(np.real(E_t))
        
    return times, np.array(energies)

def plot_relaxation():
    """Executes the simulation and plots the resulting thermalization curve."""
    N = 6         # System Size (Easily scalable to 1000 due to O(N^3) complexity)
    J = 1.0         # Interaction strength
    g = 0.5         # Transverse field strength
    beta = 2.0      # Target Inverse Temperature (Cold)
    t_max = 1000.0     # Total simulation time
    
    times, energies = solve_covariance_dynamics(N, J, g, beta, t_max)
    
    # Normalize energy per spin for standard thermodynamic plotting
    energy_per_spin = energies / N
    
    plt.figure(figsize=(8, 5))
    plt.plot(times, energy_per_spin, 'b-', linewidth=2.5, label=f"$\\beta = {beta}$")
    plt.title(f"Exact Covariance Relaxation (1D TFIM, $N={N}$)")
    plt.xlabel("Time ($t$)")
    plt.ylabel("Internal Energy per Spin ($E/N$)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Majorana_Thermalization.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_relaxation()