#DissipatorBuilder.py

import numpy as np
import scipy.linalg as la
from Ising_1D import PAULIS  # Make sure your PAULIS dict is imported

class DissipatorBuilder:
    def __init__(self, beta: float):
        self.beta = beta

    def build_ising_superoperator(self, J: float, g: float = 0.0, jump_keys: list = ['X', 'Y', 'Z']) -> np.ndarray:
        I, X, Y, Z = PAULIS['I'], PAULIS['X'], PAULIS['Y'], PAULIS['Z']

        # 3-site local physical energy neighborhood
        X0, X1, X2 = np.kron(X, np.kron(I, I)), np.kron(I, np.kron(X, I)), np.kron(I, np.kron(I, X))
        Z0, Z1, Z2 = np.kron(Z, np.kron(I, I)), np.kron(I, np.kron(Z, I)), np.kron(I, np.kron(I, Z))

        H3_local = -J * (X0 @ X1 + X1 @ X2) - g * (Z0 + Z1 + Z2)

        # Exact Diagonalization & QDB Filter
        evals, evecs = la.eigh(H3_local)
        nu_matrix = evals[:, None] - evals[None, :]
        f_matrix = np.exp(-((self.beta * nu_matrix + 1)**2) / 8.0 + 1.0/8.0)
        
        dim = 8
        super_L_local = np.zeros((dim**2, dim**2), dtype=complex)
        
        # Center Jump Operators
        center_ops = [np.kron(I, np.kron(PAULIS[key], I)) for key in jump_keys]
            
        for A in center_ops:
            # Move to energy basis, apply filter, move back
            A_eig = evecs.T.conj() @ A @ evecs
            L = evecs @ (f_matrix * A_eig) @ evecs.T.conj()
            
            L_super_jump = np.kron(L, L.conj())
            L_dag_L = L.conj().T @ L
            
            # Add purely the Dissipator components
            super_L_local += (L_super_jump 
                              - 0.5 * np.kron(L_dag_L, np.eye(dim)) 
                              - 0.5 * np.kron(np.eye(dim), L_dag_L.T))

        # We return the raw matrix. The Model class will project it!
        return super_L_local

#OpenQuantumModel.py

import itertools
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from Ising_1D import PAULIS

class OpenQuantumModel(CouplingMPOModel):
    default_lattice = Chain
    force_default_lattice = True 

    def init_sites(self, model_params):
        from Ising_1D import LiouvilleSite
        return LiouvilleSite()

    def init_terms(self, model_params):
        J = model_params.get('J', 1.0)
        g = model_params.get('g', 0.0)
        super_L = model_params.get('super_L') 
        
        # 1. Native Coherent/Hamiltonian Evolution
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'XI', u2, 'XI', dx) # Ket
            self.add_coupling( J, u1, 'IX', u2, 'IX', dx) # Bra
            
        if g != 0.0:
            self.add_onsite(-g, 0, 'ZI')
            self.add_onsite( g, 0, 'IZ')

        # 2. Pauli-String Dissipator Compiler
        self.logger.info("Compiling 3-site dissipator MPO via Pauli Strings...")
        pauli_names = ['I', 'X', 'Y', 'Z']
        
        for kets in itertools.product(pauli_names, repeat=3):
            K = np.kron(PAULIS[kets[0]], np.kron(PAULIS[kets[1]], PAULIS[kets[2]]))
            for bras in itertools.product(pauli_names, repeat=3):
                p1, p2, p3 = f"{kets[0]}{bras[0]}", f"{kets[1]}{bras[1]}", f"{kets[2]}{bras[2]}"
                
                # Trace Projector Matrix
                B = np.kron(PAULIS[bras[0]], np.kron(PAULIS[bras[1]], PAULIS[bras[2]]))
                P_123 = np.kron(K, B.T)
                
                # Calculate Hilbert-Schmidt Inner Product
                c = np.trace(P_123.conj().T @ super_L) / 64.0
                
                if abs(c) > 1e-10:
                    self.add_multi_coupling(c, [(p1, 0, 0), (p2, 1, 0), (p3, 2, 0)])

#ThermalObserver.py

import numpy as np
from tenpy.networks.mps import MPS

class ThermalObserver:
    def __init__(self, model, N):
        self.N = N
        self.model = model
        
        id_vec = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
        x_vec = np.array([0.0, 1.0, 1.0, 0.0], dtype=complex)
        z_vec = np.array([1.0, 0.0, 0.0, -1.0], dtype=complex)
        
        # 1. Identity MPS (For exact trace normalization)
        self.psi_identity = MPS.from_product_state(
            model.lat.mps_sites(), [id_vec] * N, bc=model.lat.bc_MPS, dtype=complex
        )
        
        # 2. Bond Observables for Energy (-J X_i X_i+1)
        self.xx_bonds = []
        for i in range(N):
            state_list = [id_vec] * N
            state_list[i] = x_vec
            state_list[(i+1)%N] = x_vec
            self.xx_bonds.append(MPS.from_product_state(model.lat.mps_sites(), state_list, bc=model.lat.bc_MPS, dtype=complex))
            
        # 3. Site Observables for Energy (-g Z_i)
        self.z_sites = []
        for i in range(N):
            state_list = [id_vec] * N
            state_list[i] = z_vec
            self.z_sites.append(MPS.from_product_state(model.lat.mps_sites(), state_list, bc=model.lat.bc_MPS, dtype=complex))

    def measure_energy(self, psi_t, J, g=0.0):
        # Explicitly measure and extract the real remaining Trace of the density matrix
        trace_rho = self.psi_identity.overlap(psi_t).real
        
        # Measure raw energy expectation values
        e_xx = sum([obs.overlap(psi_t).real for obs in self.xx_bonds])
        e_z = sum([obs.overlap(psi_t).real for obs in self.z_sites])
        
        raw_energy = -J * e_xx - g * e_z
        
        # Normalize the energy by the trace to fix the SVD "leakage"
        normalized_energy = (raw_energy / trace_rho) / self.N
        return trace_rho, normalized_energy
    
#run_simulation.py

import tenpy.algorithms.tdvp as tdvp
from tenpy.networks.mps import MPS
import numpy as np

def run_simulation(N=5, J=1.0, g=0.0, beta=1.0, dt=0.01, tol=1e-5):
    print(f"--- Starting Simulation: N={N}, Beta={beta} ---")
    
    # Physics Engine
    builder = DissipatorBuilder(beta=beta)
    super_L = 1j * builder.build_ising_superoperator(J, g) 
    
    # Geometry & Model Compilation
    model_params = {
        'L': N, 'J': J, 'g': g, 'conserve': 'None',
        'bc_MPS': 'finite', 'bc_x': 'periodic',
        'super_L': super_L 
    }
    model = OpenQuantumModel(model_params)
    
    # Measurement Suite
    observer = ThermalObserver(model, N)
    
    # State Initialization (Neel State [0, 3, 0, 3])
    initial_state = ['0' if i % 2 == 0 else '3' for i in range(N)]
    psi_t = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS, dtype=complex)
    
    # TDVP Engine
    tdvp_params = {
        'start_time': 0.0, 'dt': dt,
        'trunc_params': {'chi_max': 150, 'svd_min': 1e-8}
    }
    engine = tdvp.TwoSiteTDVPEngine(psi_t, model, tdvp_params)
    
    times, energies = [0.0], []
    _, e_0 = observer.measure_energy(psi_t, J, g)
    energies.append(e_0)
    
    step, stable_count = 0, 0
    min_sim_time = 2.5 
    
    while True:
        step += 1
        engine.run()
        
        _, e_t = observer.measure_energy(psi_t, J, g)
        
        times.append(engine.evolved_time)
        energies.append(e_t)
        
        if step % 10 == 0:
            print(f"Time: {engine.evolved_time:5.2f} | Energy/Bond: {e_t:.6f}")
            
        # Dynamic Termination Convergence Check
        delta_E = abs(energies[-1] - energies[-2])
        if engine.evolved_time > min_sim_time and delta_E < tol:
            stable_count += 1
        else:
            stable_count = 0
            
        if stable_count >= 10:
            print(f"\nConvergence reached at t = {engine.evolved_time:.2f}")
            break
            
        if engine.evolved_time > 100.0:
            print("\nWarning: Maximum time limit reached.")
            break
            
    return np.array(times), np.array(energies)
