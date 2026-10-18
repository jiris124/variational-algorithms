"""VQE as discretised adiabatic computation, for small hydrogen molecules in the minimal (STO-3G)
basis, for Lecture 5.

The path H(s) = (1-s) F + s H runs from the Fock operator F, whose ground state among states with the
right numbers of electrons is the Hartree-Fock determinant, to the molecular Hamiltonian H, both
written on qubits by the Jordan-Wigner transformation. Both conserve the numbers of spin-up and
spin-down electrons, so we work in that sector. With L layers, Delta t = T/L and midpoints
s_k = (k - 1/2)/L, the discretised evolution is prod_k exp(-i beta_k F) exp(-i gamma_k H) with
beta_k = Delta t (1 - s_k), gamma_k = Delta t s_k. VQE optimises the 2L angles, starting from these.

Writes, into figures/ch05/:
  adiabatic_T.dat   T and the energy error of the continuous evolution, for each molecule
  adiabatic_L.dat   L and the energy errors of the discretised evolution and of VQE, for each molecule
(errors in Hartree, above the exact ground energy), and prints the numbers quoted in the text.
"""
import math, functools, operator, os
if not hasattr(math, "prod"):                       # Python 3.7 compatibility for pyscf
    math.prod = lambda it, start=1: functools.reduce(operator.mul, it, start)
import numpy as np
import scipy.sparse as sp
from pyscf import gto, scf, fci, ao2mo
from scipy.linalg import eigh
from scipy.optimize import minimize

OUT = "figures/ch05"
os.makedirs(OUT, exist_ok=True)

def jordan_wigner(nso):
    I, Z = sp.identity(2, format="csr"), sp.csr_matrix(np.diag([1.0, -1.0]))
    lower = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))   # |0><1|: |1> is an occupied orbital
    ops = []
    for p in range(nso):
        m = sp.identity(1, format="csr")
        for q in range(nso):
            m = sp.kron(m, Z if q < p else (lower if q == p else I), format="csr")
        ops.append(m)
    return ops

def molecule(atom):
    """Return H and F restricted to the sector of the Hartree-Fock determinant, its index there,
    the exact ground energy and the Hartree-Fock energy."""
    mol = gto.M(atom=atom, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    C, nocc = mf.mo_coeff, mol.nelectron // 2
    norb = C.shape[1]; nso = 2 * norb
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), norb)          # chemists' (pq|rs)
    a = jordan_wigner(nso); ad = [x.T.tocsr() for x in a]
    H = mol.energy_nuc() * sp.identity(2**nso, format="csr")
    for p in range(nso):                             # spin orbital p = 2 i + spin
        for q in range(nso):
            if p % 2 == q % 2 and abs(h1[p // 2, q // 2]) > 1e-12:
                H = H + h1[p // 2, q // 2] * (ad[p] @ a[q])
    pairs = {(p, q): ad[p] @ a[q] for p in range(nso) for q in range(nso)}
    for p, q, r, s in np.ndindex(nso, nso, nso, nso):
        if p % 2 == q % 2 and r % 2 == s % 2 and p != r:
            v = eri[p // 2, q // 2, r // 2, s // 2]
            if abs(v) > 1e-12:
                H = H + 0.5 * v * (ad[p] @ ad[r] @ a[s] @ a[q])
    F = sum(mf.mo_energy[p // 2] * pairs[(p, p)] for p in range(nso))
    occ = np.array([[(x >> (nso - 1 - p)) & 1 for p in range(nso)] for x in range(2**nso)])
    sector = np.where((occ[:, 0::2].sum(1) == nocc) & (occ[:, 1::2].sum(1) == nocc))[0]
    H = H.toarray()[np.ix_(sector, sector)]; F = F.toarray()[np.ix_(sector, sector)]
    hf = int(np.where(sector == int("1" * (2 * nocc) + "0" * (nso - 2 * nocc), 2))[0][0])
    F = F + (H[hf, hf] - F[hf, hf]) * np.eye(len(sector))      # so that <HF|F|HF> = E_HF
    e_fci = fci.FCI(mf).kernel()[0]
    w = eigh(H, eigvals_only=True)
    assert abs(w[0] - e_fci) < 1e-8 and abs(H[hf, hf] - mf.e_tot) < 1e-8
    assert np.allclose(F, np.diag(np.diag(F)))
    return H, np.real(np.diag(F)), hf, w[0], mf.e_tot, len(sector)

class Path:
    def __init__(self, H, f, hf):
        self.H, self.f = H, f
        self.w, self.U = eigh(H)
        self.psi0 = np.zeros(len(f), complex); self.psi0[hf] = 1
    def energy(self, psi): return float(np.real(psi.conj() @ self.H @ psi))
    def layers(self, beta, gamma):
        psi = self.psi0
        for b, g in zip(beta, gamma):
            psi = self.U @ (np.exp(-1j * g * self.w) * (self.U.conj().T @ psi))
            psi = np.exp(-1j * b * self.f) * psi
        return psi
    def continuous(self, T, steps=3000):
        psi = self.psi0; dt = T / steps
        for k in range(steps):
            s = (k + 0.5) / steps
            w, U = eigh((1 - s) * np.diag(self.f) + s * self.H)
            psi = U @ (np.exp(-1j * dt * w) * (U.conj().T @ psi))
        return psi

def chain(n, d): return "; ".join(f"H 0 0 {k * d}" for k in range(n))
MOLECULES = [("H2, 0.75", chain(2, 0.75)), ("H4, 1.0", chain(4, 1.0)), ("H4, 1.5", chain(4, 1.5))]
TS = [1, 2, 4, 8, 16, 32, 64]
T_FIXED, LS = 16.0, list(range(1, 13))

resT, resL = {}, {}
for name, atom in MOLECULES:
    H, f, hf, e0, ehf, dim = molecule(atom)
    P = Path(H, f, hf)
    print(f"{name}: sector dimension {dim}, E_HF = {ehf:.6f}, E_0 = {e0:.6f}, correlation {ehf - e0:.4f} Ha,"
          f" ||[F,H]|| = {np.linalg.norm(np.diag(f) @ H - H @ np.diag(f), 2):.3f}")
    resT[name] = [P.energy(P.continuous(T)) - e0 for T in TS]
    print("   continuous: " + ", ".join(f"T={T}: {e:.1e}" for T, e in zip(TS, resT[name])))
    resL[name] = []
    for L in LS:
        dt = T_FIXED / L; s = (np.arange(1, L + 1) - 0.5) / L
        beta, gamma = dt * (1 - s), dt * s
        e_disc = P.energy(P.layers(beta, gamma)) - e0
        res = minimize(lambda x: P.energy(P.layers(x[:L], x[L:])), np.concatenate([beta, gamma]),
                       method="BFGS", options={"gtol": 1e-10})
        resL[name].append((e_disc, max(res.fun - e0, 1e-12)))
        print(f"   T = {T_FIXED}, L = {L:2d}: discretised {e_disc:.2e}, VQE from there {res.fun - e0:.2e}")

with open(f"{OUT}/adiabatic_T.dat", "w") as fh:
    fh.write("T " + " ".join(f"m{i}" for i in range(len(MOLECULES))) + "\n")
    for k, T in enumerate(TS):
        fh.write(f"{T} " + " ".join(f"{max(resT[n][k], 1e-12):.6e}" for n, _ in MOLECULES) + "\n")
with open(f"{OUT}/adiabatic_L.dat", "w") as fh:
    fh.write("L " + " ".join(f"disc{i} vqe{i}" for i in range(len(MOLECULES))) + "\n")
    for k, L in enumerate(LS):
        fh.write(f"{L} " + " ".join(f"{max(resL[n][k][0], 1e-12):.6e} {resL[n][k][1]:.6e}" for n, _ in MOLECULES) + "\n")
