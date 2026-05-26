import numpy as np
from tenpy.networks.mps import MPS

class ThermalObserver:
    """Handles superspace trace normalization and exact observable extraction."""
    
    def __init__(self, model, N: int):
        self.model = model
        self.N = N
        self.id_mps = self._build_identity_mps()
        self.bond_observables = self._build_bond_observables()

    def _build_identity_mps(self):
        # [1, 0, 0, 1] represents the trace in Liouville space
        id_vec = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
        return MPS.from_product_state(
            self.model.lat.mps_sites(), [id_vec] * self.N, 
            bc=self.model.lat.bc_MPS, dtype=complex
        )

    def _build_bond_observables(self):
        id_vec = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
        x_vec = np.array([0.0, 1.0, 1.0, 0.0], dtype=complex)
        observables = []
        for i in range(self.N):
            state_list = [id_vec] * self.N
            state_list[i] = x_vec
            state_list[(i+1) % self.N] = x_vec
            obs_mps = MPS.from_product_state(
                self.model.lat.mps_sites(), state_list, 
                bc=self.model.lat.bc_MPS, dtype=complex
            )
            observables.append(obs_mps)
        return observables

    def measure_energy(self, psi_t, J: float):
        """Returns the L1 Trace and the normalized internal energy."""
        trace_rho = self.id_mps.overlap(psi_t).real
        raw_energy = sum([-J * obs.overlap(psi_t).real for obs in self.bond_observables])
        normalized_energy = (raw_energy / trace_rho) / self.N
        return trace_rho, normalized_energy