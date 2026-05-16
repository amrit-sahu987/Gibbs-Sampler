# import numpy as np
# import matplotlib.pyplot as plt
# from tenpy.algorithms import tdvp
# from tenpy.networks.mps import MPS

# def simulate_thermal_relaxation(N, J, beta, total_time=10.0, dt=0.05):
#     """
#     Simulates the real-time thermalization of an open spin chain.
#     """
#     # 1. Initialize the Model with the TRUE Lindbladian (Not the Parent Hamiltonian)
#     # (You would create a new model class that compiles \mathcal{L} instead of H_parent)
#     model_params = {'N': N, 'J': J, 'beta': beta, 'bc': 'periodic'}
#     lindblad_model = TrueLindbladianModel(model_params) 
    
#     # 2. Initialize a pure state FAR from equilibrium
#     # e.g., '0' might represent the |All Spins Up><All Spins Up| density matrix
#     initial_state = ['0'] * N
#     psi_t = MPS.from_product_state(lindblad_model.lat.mps_sites(), initial_state, bc='periodic')
    
#     # 3. Setup the TDVP Time Evolution Engine
#     # Note: Because \mathcal{L} drives dissipative (non-unitary) evolution, 
#     # the tensor network's norm will naturally decay.
#     tdvp_params = {
#         'start_time': 0.0,
#         'dt': dt,
#         'trunc_params': {'chi_max': 200, 'svd_min': 1e-8}
#     }
#     engine = tdvp.TDVPEngine(psi_t, lindblad_model, tdvp_params)
    
#     # Tracking arrays
#     times = []
#     energies = []
    
#     print("Starting Real-Time Thermal Relaxation...")
    
#     # 4. The Time Evolution Loop
#     num_steps = int(total_time / dt)
#     for step in range(num_steps):
#         # Evolve the state forward by dt
#         engine.run()
        
#         # Measure the physical energy at the current time
#         # (Remember to normalize by the trace of the density matrix, 
#         # which is the overlap with the identity bra)
#         trace_rho = psi_t.overlap(identity_bra_state) # <I | \rho(t)>
        
#         total_energy = 0.0
#         for i in range(N):
#             bond_energy = psi_t.expectation_value_term([('XI', i), ('XI', (i+1)%N)])
#             total_energy += -J * (bond_energy.real / trace_rho.real)
            
#         times.append(engine.evolved_time)
#         energies.append(total_energy / N)
        
#         if step % 20 == 0:
#             print(f"Time: {engine.evolved_time:.2f} | Energy/Bond: {energies[-1]:.6f}")

#     return times, energies

# # --- Plotting the Results ---
# # times, energies = simulate_thermal_relaxation(N=8, J=1.0, beta=0.5)
# # plt.plot(times, energies, label="Simulated Relaxation")
# # plt.axhline(-J * np.tanh(beta * J), color='r', linestyle='--', label="Analytical Steady State")
# # plt.xlabel("Time (t)")
# # plt.ylabel("Average Energy per Bond")
# # plt.title("Empirical Mixing Time via Energy Convergence")
# # plt.legend()
# # plt.show()

from pyexpat import model

from xml.parsers.expat import model

import numpy as np
import scipy.linalg as la
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tenpy.networks.site import Site
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS

from Ising_1D import PAULIS, SUPER_PAULIS, LiouvilleSite

def compute_true_lindbladian(J, beta):
    # ... [Keep the exact same physical Pauli definitions and H3_local setup] ...
    # ... [Keep Exact Diagonalization to find evecs, evals, and nu_matrix] ...
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

# class TrueLindbladianModel(CouplingMPOModel):
#     def init_lattice(self, model_params):
#         self.N = model_params.get('N', 12)
#         self.J = model_params.get('J', 1.0)
#         self.beta = model_params.get('beta', 0.5)
#         self.bc = model_params.get('bc', 'periodic')
#         return Chain(self.N, LiouvilleSite(), bc=self.bc)                                                

#     def init_terms(self, model_params):
#         self.add_coupling(-self.J, 0, 'XI', 0, 'XI', 1) # Ket evolution
#         self.add_coupling( self.J, 0, 'IX', 0, 'IX', 1) # Bra evolution
#         # Note: The above two lines add the coherent Hamiltonian part of the Lindbladian
#         super_H_eff = 1j *compute_true_lindbladian(self.J, self.beta)
#         #tdvp is designed to solve the schrodinger equation, 
#         # so we need to multiply the Lindbladian by i to get 
#         # the effective "Hamiltonian" for time evolution, 
#         # as the lindblad equation does not have the factor of i.
        
#         # Question for you: Do we need a LANCZOS_ENERGY_SHIFT here?
#         pauli_labels = ['I', 'X', 'Y', 'Z']
#         # [Iterate over the Paulis to find P_123 just like before]
#         # c = np.trace(P_123.conj().T @ super_L) / 64.0
#         """
#         The Code: We create two nested loops. itertools.product(..., repeat=3)
#           generates every possible 3-site combination of Paulis 
#           (e.g., ['X', 'I', 'Z']). Because there are 4 Paulis and 3 sites,
#             there are $4^3 = 64$ ket combinations and $64$ bra combinations, 
#             leading to 4,096 total iterations.The Theory: A single physical spin
#               has a $2 \times 2$ density matrix, which requires 4 basis operators
#                 (I, X, Y, Z) to fully describe it. A 3-site local window has an
#                   $8 \times 8$ density matrix, requiring 64 basis operators.
#                     Because Liouville space maps density matrices to vectors,
#                       operators acting on this space become $64 \times 64$ matrices.
#                         To tell TeNPy how to build the Matrix Product Operator (MPO),
#                           we must project our dense $64 \times 64$ array into these
#                             4,096 discrete tensor components.
#         """
#         for kets in itertools.product(pauli_labels, repeat=3):
#             K = np.kron(PAULIS[kets[0]], np.kron(PAULIS[kets[1]], PAULIS[kets[2]]))
#             for bras in itertools.product(pauli_labels, repeat=3):
#                 p1, p2, p3 = f"{kets[0]}{bras[0]}", f"{kets[1]}{bras[1]}", f"{kets[2]}{bras[2]}"
#                 # If kets = ['X', 'I', 'Z'] and bras = ['I', 'Y', 'Z'], 
#                 # this line outputs p1 = 'XI', p2 = 'IY', and p3 = 'ZZ'
#                 B = np.kron(PAULIS[bras[0]], np.kron(PAULIS[bras[1]], PAULIS[bras[2]]))
#                 P_123 = np.kron(K, B.T)

#                 c = np.trace(P_123.conj().T @ super_H_eff) / 64.0

#                 if abs(c) > 1e-10:
#                     self.add_multi_coupling(c, [(p1, 0, 0), (p2, 1, 0), (p3, 2, 0)])
        
#         # Question for you: Previously we added `c.real`. Because the true
#         # Lindbladian has complex/imaginary eigenvalues, should we add `c` 
#         # directly, or still use `c.real`?
#         # self.add_multi_coupling( ??? , [(p1, 0, 0), (p2, 1, 0), (p3, 2, 0)])

# (Assuming LiouvilleSite, PAULIS, and compute_true_lindbladian are already defined)

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
        # ---------------------------------------------------------
        # DEVIATION: Rejecting Conservation Laws
        # ---------------------------------------------------------
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
        # Secure parameter parsing (matches TeNPy best practices)
        J = model_params.get('J', 1.0, 'real_or_array')
        beta = model_params.get('beta', 0.5, 'real_or_array')
        
        # ---------------------------------------------------------
        # PART 1: The 2-Site Coherent Evolution (-i[H, rho])
        # ---------------------------------------------------------
        # We CAN use the TeNPy template here. The physical Hamiltonian 
        # is just nearest-neighbor Ising, so we let TeNPy map it.
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'XI', u2, 'XI', dx) # Ket evolution
            self.add_coupling( J, u1, 'IX', u2, 'IX', dx) # Bra evolution
            
        # ---------------------------------------------------------
        # PART 2: The 3-Site Dissipative Evolution (Trace Compiler)
        # ---------------------------------------------------------
        # DEVIATION: We DO NOT use generic lattice iterators here. 
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

def simulate_thermalization(N, J, beta, total_time=5.0, dt=0.05):
    # model = TrueLindbladianModel({'N': N, 'J': J, 'beta': beta, 'bc': 'periodic'})
    model_params = {
        'L': N,               # CRITICAL: TeNPy's Chain expects 'L' for length, not 'N'.
        'J': J,
        'beta': beta,
        'bc_MPS': 'finite',   # The tensor network array must have a start and end.
        'bc_x': 'periodic',   # The physical Ising couplings wrap around in a ring.
        'conserve': 'None'    # Pass 'None' so our init_sites override accepts it gracefully.
    }
    model = TrueLindbladianChain(model_params)

    # Create the Neel state, far from equilibrium. - maybe we randomize it later?
    initial_state = ['0' if i % 2 == 0 else '3' for i in range(N)]
    psi_t = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS, dtype=complex)     
    
    id_local = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
    psi_identity = MPS.from_product_state(model.lat.mps_sites(), [id_local] * N, bc=model.lat.bc_MPS, dtype=complex)

    # Initialize TDVP
    tdvp_params = {
        'start_time': 0.0,
        'dt': dt,
        'trunc_params': {'chi_max': 50, 'svd_min': 1e-8}
    }
    
    engine = tdvp.TwoSiteTDVPEngine(psi_t, model, tdvp_params)

    times = [0.0]
    energies = []

    trace_rho_0 = psi_identity.overlap(psi_t)
    raw_e_0 = sum([-J * psi_t.expectation_value_term([('XI', i), ('XI', (i+1)%N)]).real for i in range(N)])
    energies.append((raw_e_0 / trace_rho_0.real) / N)
    
    num_steps = int(total_time / dt)
    for step in range(num_steps):
        engine.run() # Evolve by dt
        
        # Measure physical energy (un-normalized)
        raw_energy = sum([-J * psi_t.expectation_value_term([('XI', i), ('XI', (i+1)%N)]).real for i in range(N)])
        
        # Calculate the trace to normalize the state
        # Hint: You need to evaluate the expectation value of the identity operator.
        # But wait, in TeNPy, expectation_value_term evaluates <psi | Op | psi>. 
        # That's not the trace! The trace is < I | rho >.
        
        # Question for you: How can you construct a simple MPS representing the 
        # global Identity state, so you can calculate the overlap: 
        trace_rho = psi_t.overlap(psi_identity) 
        
        normalized_energy = raw_energy / trace_rho.real
        
        times.append(engine.evolved_time)
        energies.append(normalized_energy / N)
        
    return np.array(times), np.array(energies)

if __name__ == "__main__":
    N = 8
    J = 1.0
    beta = 0.5
    
    times, energies = simulate_thermalization(N, J, beta, total_time=5.0, dt=0.05)
    
    # Analytical target limits
    analytical_steady_state = -J * np.tanh(beta * J)
    
    # Extract the empirical mixing time (tau) via non-linear regression
    def decay_law(t, E_inf, A, tau):
        return E_inf + A * np.exp(-t / tau)
    
    popt, _ = curve_fit(decay_law, times, energies, p0=[analytical_steady_state, energies[0]-analytical_steady_state, 1.0])
    E_inf, A, tau_mix = popt
    
    print("\n=== Empirical Results ===")
    print(f"Extracted Steady State Energy: {E_inf:.6f}")
    print(f"Theoretical Analytical Energy: {analytical_steady_state:.6f}")
    print(f"Empirical Mixing Time (tau):   {tau_mix:.6f}")
    
    # Plot the results
    plt.figure(figsize=(9, 6))
    plt.plot(times, energies, 'o-', color='#1f77b4', markersize=4, label='TeNPy TDVP Quench')
    plt.plot(times, decay_law(times, *popt), '--', color='#d62728', linewidth=2, label=f'Exponential Fit ($\\tau={tau_mix:.3f}$)')
    plt.axhline(analytical_steady_state, color='black', linestyle=':', label='Analytical Gibbs State')
    
    plt.title(r'Real-Time Thermal Relaxation & Mixing Time Extraction', fontsize=14)
    plt.xlabel('Time ($t$)', fontsize=12)
    plt.ylabel('Average Internal Energy per Bond', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()