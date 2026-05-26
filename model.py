from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite 
import numpy as np
import itertools
import tenpy

from builder import PAULIS

from tenpy.networks.site import Site
from tenpy.linalg.charges import LegCharge
import numpy as np

class LiouvilleSite(Site):
    """Custom d=4 site for open quantum systems in Liouville superspace."""
    def __init__(self, conserve='None'):
        # Liouville space for a single spin-1/2 has local dimension d=4
        leg = LegCharge.from_trivial(4)
        
        ops = {}
        # build all 16 superspace operators (e.g., 'XI', 'IX', 'ZZ')
        for name1, mat1 in PAULIS.items():
            for name2, mat2 in PAULIS.items():
                op_name = f"{name1}{name2}"
                # Kronecker product matches the Choi-Jamiolkowski vectorization order
                ops[op_name] = np.kron(mat1, mat2)
                        
        # Initialize the TeNPy Site with these custom operators
        super().__init__(leg, state_labels=None, **ops)

class OpenQuantumModel(CouplingMPOModel):
    """System-agnostic open quantum model for Liouville space."""
    
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'None')
        if conserve != 'None':
            self.logger.warning(f"Overriding conserve='{conserve}' to 'None'.")
        
        # Use our custom d=4 supersite
        site = LiouvilleSite(conserve='None') 
        return site

    def init_terms(self, model_params):
        # Explicitly provide the default for J 
        J = model_params.get('J', 1.0)
        
        super_L = model_params['super_L']
        stencil = model_params['stencil']
        
        # Coherent Evolution (2-Site)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'XI', u2, 'XI', dx) # Ket
            self.add_coupling( J, u1, 'IX', u2, 'IX', dx) # Bra
            
        # Dissipative Evolution (Mapped via dynamic stencil)
        pauli_names = ['I', 'X', 'Y', 'Z']
        for kets in itertools.product(pauli_names, repeat=3):
            K = np.kron(PAULIS[kets[0]], np.kron(PAULIS[kets[1]], PAULIS[kets[2]]))
            for bras in itertools.product(pauli_names, repeat=3):
                p_str = [f"{kets[i]}{bras[i]}" for i in range(3)]
                B = np.kron(PAULIS[bras[0]], np.kron(PAULIS[bras[1]], PAULIS[bras[2]]))
                P_123 = np.kron(K, B.T)
                
                # Trace projection
                c = np.trace(P_123.conj().T @ super_L) / 64.0
                
                if abs(c) > 1e-10:
                    # Dynamically map the 3 operators to the spatial stencil
                    couplings = [(p_str[i], stencil[i][0], stencil[i][1]) for i in range(len(stencil))]
                    self.add_multi_coupling(c, couplings)