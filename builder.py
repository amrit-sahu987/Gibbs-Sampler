import numpy as np
import scipy.linalg as la

PAULIS = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    'Sp': np.array([[0, 1], [0, 0]], dtype=complex), 
    'Sm': np.array([[0, 0], [1, 0]], dtype=complex)
}

class DissipatorBuilder:
    """Compiles local Hamiltonians into detailed-balanced Liouvillian superoperators."""
    
    def __init__(self, beta: float):
        self.beta = beta

    def build_ising_superoperator(self, J: float, g: float = 0.0, jump_keys: list = ['X', 'Y', 'Z']) -> np.ndarray:
        """Builds the 64x64 non-Hermitian dissipator matrix for the 1D Ising model."""
        I, X, Y, Z, Sp, Sm = PAULIS['I'], PAULIS['X'], PAULIS['Y'], PAULIS['Z'], PAULIS['Sp'], PAULIS['Sm']

        # 3-site local physical energy neighborhood
        X0 = np.kron(X, np.kron(I, I))
        X1 = np.kron(I, np.kron(X, I))
        X2 = np.kron(I, np.kron(I, X))
        # H3_local = -J * (X0 @ X1 + X1 @ X2) for zero field

        Z0 = np.kron(Z, np.kron(I, I))
        Z1 = np.kron(I, np.kron(Z, I))
        Z2 = np.kron(I, np.kron(I, Z))

        H3_local = -J * (X0 @ X1 + X1 @ X2) - g * (Z0 + Z1 + Z2) #TFIM with external field

        # Exact Diagonalization
        evals, evecs = la.eigh(H3_local)
        nu_matrix = evals[:, None] - evals[None, :]

        # QDB Filter
        f_matrix = np.exp(-((self.beta * nu_matrix + 1)**2) / 8.0 + 1.0/8.0)

        # Construct Jump Operators on the Central Spin
        # center_ops = [np.kron(I, np.kron(op, I)) for op in [X, Y, Z]]
        center_ops = []
        for key in jump_keys:
            if key in PAULIS:
                center_ops.append(np.kron(I, np.kron(PAULIS[key], I)))
            else:
                raise ValueError(f"Invalid jump operator key: {key}")
            
        dim = 8
        super_L_local = np.zeros((dim**2, dim**2), dtype=complex)

        # True Lindbladian Assembly in Liouville Superspace
        for A in center_ops:
            # Transform central operator into energy eigenbasis, apply filter, transform back
            A_eig = evecs.T.conj() @ A @ evecs
            L = evecs @ (f_matrix * A_eig) @ evecs.T.conj()
            
            L_dag_L = L.conj().T @ L
            
            # Vectorize: L(rho) = L * rho * L^dag - 0.5 * L^dag * L * rho - 0.5 * rho * L^dag * L
            # Using convention: A * rho * B --> np.kron(A, B.T)
            term_jump = np.kron(L, L.conj()) 
            term_anti1 = -0.5 * np.kron(L_dag_L, np.eye(dim))
            term_anti2 = -0.5 * np.kron(np.eye(dim), L_dag_L.T)
            
            super_L_local += (term_jump + term_anti1 + term_anti2)

        return super_L_local