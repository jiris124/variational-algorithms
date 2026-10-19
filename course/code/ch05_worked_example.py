"""The worked example of Lecture 5: depth-one QAOA for MaxCut on the 4-cycle, simulated gate by
gate (Hadamards, CNOT - R_Z - CNOT for each edge, R_X for the mixer), with every energy estimated
from M measured copies and every gradient from the parameter-shift rule, gate by gate.

Conventions as in Lecture 6: H_C = -1/2 sum_e (1 - Z_i Z_j), H_M = sum_j X_j,
|psi(beta, gamma)> = exp(-i beta H_M) exp(-i gamma H_C) |+>^4, minimise F = <H_C> = -<cut>.
Up to a global phase, exp(-i gamma H_C) = prod_e R_ZZ(gamma) and exp(-i beta H_M) = prod_j R_X(2 beta).

Writes, into figures/ch05/:
  worked_shots.dat  counts of the cut values 0, 2, 4 among the M copies at the starting point
  worked_path.dat   the iterates: k beta gamma F_hat se F_exact g_beta g_gamma g_beta_exact g_gamma_exact
and prints the numbers quoted on the slides.
"""
import os
import numpy as np

OUT = "figures/ch05"
os.makedirs(OUT, exist_ok=True)
n, edges = 4, [(0, 1), (1, 2), (2, 3), (3, 0)]
M, eta, steps = 1000, 0.1, 6
rng = np.random.default_rng(2026)

# --------------------------------------------------------------------------- gates
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
def rz(t): return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
def rx(t): return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])

def apply_1q(psi, U, q):
    """Apply the 2x2 matrix U to qubit q (qubit 0 is the leftmost bit)."""
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)

def apply_cnot(psi, c, t):
    psi = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[c] = 1
    sub = psi[tuple(idx)]                      # the half with control = 1
    t2 = t if t < c else t - 1
    psi[tuple(idx)] = np.flip(sub, axis=t2)    # flip the target there
    return psi.reshape(-1)

def circuit(theta_zz, theta_x):
    """theta_zz[e]: angle of the R_Z in the CNOT-R_Z-CNOT of edge e; theta_x[j]: angle of R_X on qubit j."""
    psi = np.zeros(2**n, complex); psi[0] = 1
    for q in range(n):
        psi = apply_1q(psi, H, q)
    for (i, j), t in zip(edges, theta_zz):
        psi = apply_cnot(psi, i, j)
        psi = apply_1q(psi, rz(t), j)
        psi = apply_cnot(psi, i, j)
    for q, t in enumerate(theta_x):
        psi = apply_1q(psi, rx(t), q)
    return psi

bits = np.array([[(x >> (n - 1 - q)) & 1 for q in range(n)] for x in range(2**n)])
cuts = np.array([sum(b[i] != b[j] for i, j in edges) for b in bits])

def exact(theta_zz, theta_x):
    p = np.abs(circuit(theta_zz, theta_x))**2
    return -(p @ cuts)

def sample(theta_zz, theta_x, shots):
    """Prepare and measure `shots` copies; return the measured bit strings (as integers)."""
    p = np.abs(circuit(theta_zz, theta_x))**2
    return rng.choice(2**n, size=shots, p=p / p.sum())

def estimate(theta_zz, theta_x, shots=M):
    c = -cuts[sample(theta_zz, theta_x, shots)]
    return c.mean(), c.std(ddof=1) / np.sqrt(shots)

def angles(beta, gamma):
    return np.full(len(edges), gamma), np.full(n, 2 * beta)

def F_closed(beta, gamma): return -2 + np.sin(4 * beta) * np.sin(2 * gamma)

# check the closed form against the gate-level simulation
for b, g in [(0.1, 0.3), (0.7, -0.4), (np.pi / 8, -np.pi / 4)]:
    assert abs(exact(*angles(b, g)) - F_closed(b, g)) < 1e-12

def psr_gradient(beta, gamma, shots=M, record=None):
    """Gradient of F by the parameter-shift rule, one gate at a time: 2|E| + 2n circuits."""
    tz, tx = angles(beta, gamma)
    g_gamma = 0.0
    for e in range(len(edges)):
        plus, minus = tz.copy(), tz.copy()
        plus[e] += np.pi / 2; minus[e] -= np.pi / 2
        fp, fm = estimate(plus, tx, shots)[0], estimate(minus, tx, shots)[0]
        g_gamma += 0.5 * (fp - fm)                       # d gamma_e / d gamma = 1
        if record is not None: record.append(("gamma", e, fp, fm, exact(plus, tx), exact(minus, tx)))
    g_beta = 0.0
    for q in range(n):
        plus, minus = tx.copy(), tx.copy()
        plus[q] += np.pi / 2; minus[q] -= np.pi / 2
        fp, fm = estimate(tz, plus, shots)[0], estimate(tz, minus, shots)[0]
        g_beta += 2 * 0.5 * (fp - fm)                    # the R_X angle is 2 beta
        if record is not None: record.append(("beta", q, fp, fm, exact(tz, plus), exact(tz, minus)))
    return g_beta, g_gamma

def exact_gradient(beta, gamma):
    return 4 * np.cos(4 * beta) * np.sin(2 * gamma), 2 * np.sin(4 * beta) * np.cos(2 * gamma)

# --------------------------------------------------------------------------- the run
beta, gamma = 0.10, -0.10
shots0 = sample(*angles(beta, gamma), M)
print("first twelve copies at the start:", " ".join(f"{x:04b}:{cuts[x]}" for x in shots0[:12]))
counts = {c: int(np.sum(cuts[shots0] == c)) for c in (0, 2, 4)}
p_exact = np.abs(circuit(*angles(beta, gamma)))**2
probs = {c: float(p_exact[cuts == c].sum()) for c in (0, 2, 4)}
with open(f"{OUT}/worked_shots.dat", "w") as f:
    f.write("cut count prob\n")
    for c in (0, 2, 4):
        f.write(f"{c} {counts[c]} {probs[c]:.6f}\n")
print("counts of cut values among", M, "copies:", counts, " exact probabilities:", {c: round(p, 4) for c, p in probs.items()})
v = -cuts[shots0]
print(f"estimate F = {v.mean():.4f} +- {v.std(ddof=1)/np.sqrt(M):.4f}, exact {F_closed(beta, gamma):.4f}")

rows = []
for k in range(steps + 1):
    Fh, se = (v.mean(), v.std(ddof=1) / np.sqrt(M)) if k == 0 else estimate(*angles(beta, gamma))
    rec = [] if k == 0 else None
    gb, gg = psr_gradient(beta, gamma, record=rec)
    gbe, gge = exact_gradient(beta, gamma)
    rows.append((k, beta, gamma, Fh, se, F_closed(beta, gamma), gb, gg, gbe, gge))
    if k == 0:
        print("shifted circuits at the start (parameter, gate, F+ hat, F- hat, F+ exact, F- exact):")
        for r in rec:
            print(f"  {r[0]:5s} {r[1]}  {r[2]:+.3f} {r[3]:+.3f}   {r[4]:+.4f} {r[5]:+.4f}")
    if k < steps:
        beta, gamma = beta - eta * gb, gamma - eta * gg
with open(f"{OUT}/worked_path.dat", "w") as f:
    f.write("k beta gamma F se Fexact gbeta ggamma gbetaexact ggammaexact\n")
    for r in rows:
        f.write(" ".join(f"{x:.6f}" if i else str(x) for i, x in enumerate(r)) + "\n")
print(" k   beta   gamma    F_hat    se     F_exact  g_beta  g_gamma  (exact)")
for r in rows:
    print(f" {r[0]}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:+.3f}  {r[4]:.3f}  {r[5]:+.3f}  {r[6]:+.3f}  {r[7]:+.3f}  ({r[8]:+.3f} {r[9]:+.3f})")
pf = np.abs(circuit(*angles(beta, gamma)))**2
final = sample(*angles(beta, gamma), M)
print(f"at the last iterate: probability of a maximum cut {pf[cuts == 4].sum():.3f}; maximum cuts among {M} copies: {np.sum(cuts[final] == 4)}")
print(f"optimum: F = -3 at beta = pi/8 = {np.pi/8:.3f}, gamma = -pi/4 = {-np.pi/4:.3f}")
print(f"circuits per iteration: 1 + 2*{len(edges)} + 2*{n} = {1 + 2*len(edges) + 2*n}; copies per iteration: {(1 + 2*len(edges) + 2*n)*M}")
