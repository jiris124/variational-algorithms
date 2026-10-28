"""Numerical data for the figures and the numbers quoted in Lecture 6 (QAOA).

Writes, into figures/ch06/:
  crossing_spectrum.dat  s and the four lowest eigenvalues of H(s) = -(1-s) sum_j X_j + s H_C
                         on the full space of 8 qubits, for the MaxCut instance of Lecture 4
  depth_one.dat          D and the best expected cut fraction of depth-one QAOA on D-regular
                         triangle-free graphs, 1/2 + max_gamma (1/2) sin(gamma) cos(gamma)^(D-1)
  depth_one_instances.dat   D and, on 20 random D-regular graphs with 22 vertices, the mean and standard
                         deviation of the cut fraction of one run of depth-one QAOA at the best angles of the
                         instance and of one Goemans-Williamson rounding, and of the maximum cut fraction
  depth_one_bands.dat    the bands of one standard deviation about these means, as closed polygons
  proof_maxcut.dat       L and the distance from the optimal subspace of QAOA of depth L at the angles
                         of the midpoint schedule, for several T, on the MaxCut instance of Lecture 4
  proof_maxcut_adiabatic.dat   T and the distance of the continuous adiabatic evolution
  proof_bisection.dat    the same for the maximum bisection of the triangular prism with the ring XY
                         mixer, simultaneous and sequential (even edges, then odd edges)
  proof_bisection_adiabatic.dat   T and the distance of the continuous evolution
  interp_angles.dat      the angles found by the interpolation heuristic of Zhou et al. at depths 2, 4, ..., 10
                         on the MaxCut instance of Lecture 4, in columns x<p> = (i-1)/(p-1), b<p> = beta_i and
                         g<p> = -gamma_i, i = 1, ..., p (nan below)
  interp_ratio.dat       p and one minus the approximation ratio of the heuristic, and of local optimisation from
                         20 random starts (median and best)
  mixer_action.dat       i and the probabilities of the 16 bit strings on 4 qubits before and after the mixer
                         e^{-i beta H_M}, beta = pi/8: (a) the transverse field from |0000>, (b) the ring XY mixer
                         from |0011>, (c) depth-one QAOA on the 4-cycle at gamma = -pi/4, from e^{-i gamma H_C}|+>^4;
                         the cut of each string, and the phase (in degrees) of its amplitude in (c) before the mixer

and checks, by statevector simulation in the conventions of Lecture 5, the closed form
F_1(beta, gamma) / |E| = -1/2 + (1/2) sin(4 beta) sin(gamma) cos(gamma)^(D-1)
on the 4-cycle (D = 2), the cube (D = 3) and the Petersen graph (D = 3).
"""
import itertools, os
import numpy as np
from scipy.linalg import eigh, expm

OUT = "figures/ch06"
os.makedirs(OUT, exist_ok=True)

def cut_diagonal(n, edges):
    z = np.array(list(itertools.product([0, 1], repeat=n)))
    return -np.sum([(z[:, u] != z[:, v]) for u, v in edges], axis=0).astype(float)

def x_sum(n):
    """Sum_j X_j as a dense matrix (n small)."""
    dim = 2 ** n
    M = np.zeros((dim, dim))
    for x in range(dim):
        for j in range(n):
            M[x ^ (1 << j), x] += 1.0
    return M

def qaoa_one(n, hc, beta, gamma):
    """F_1 = <psi|H_C|psi> with psi = e^{-i beta H_M} e^{-i gamma H_C} |+>^n."""
    psi = np.full(2 ** n, 2 ** (-n / 2), complex) * np.exp(-1j * gamma * hc)
    for j in range(n):                            # e^{-i beta X_j} on each qubit
        psi = psi.reshape(2 ** (n - 1 - j), 2, 2 ** j)
        a, b = psi[:, 0, :].copy(), psi[:, 1, :].copy()
        psi[:, 0, :] = np.cos(beta) * a - 1j * np.sin(beta) * b
        psi[:, 1, :] = np.cos(beta) * b - 1j * np.sin(beta) * a
        psi = psi.reshape(-1)
    return float(np.real(np.vdot(psi, hc * psi)))

# ------------------------------------------------------------ the depth-one closed form
graphs = {
    "4-cycle": (4, [(0, 1), (1, 2), (2, 3), (3, 0)], 2),
    "cube": (8, [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)], 3),
    "Petersen": (10, [(i, (i + 1) % 5) for i in range(5)] + [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
                 + [(i, i + 5) for i in range(5)], 3),
}
rng = np.random.default_rng(6)
for name, (n, edges, D) in graphs.items():
    hc = cut_diagonal(n, edges)
    err = 0.0
    for beta, gamma in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        closed = len(edges) * (-0.5 + 0.5 * np.sin(4 * beta) * np.sin(gamma) * np.cos(gamma) ** (D - 1))
        err = max(err, abs(qaoa_one(n, hc, beta, gamma) - closed))
    print(f"{name}: n = {n}, |E| = {len(edges)}, D = {D}, max deviation from the closed form {err:.1e}")

g3 = np.arctan(1 / np.sqrt(2))
print(f"D = 3: best cut fraction 1/2 + (1/2) sin g cos^2 g at g = arctan(1/sqrt 2) = {0.5 + 0.5 * np.sin(g3) * np.cos(g3) ** 2:.5f}")
rows = []
for D in range(2, 51):
    g = np.arctan(1 / np.sqrt(D - 1))
    rows.append((D, 0.5 + 0.5 * np.sin(g) * np.cos(g) ** (D - 1), 0.5 + 0.5 * np.exp(-0.5) / np.sqrt(D)))
np.savetxt(f"{OUT}/depth_one.dat", rows, fmt="%.6f", header="D best asymptotic", comments="")
print(f"large D: (1/2) e^(-1/2) = {0.5 * np.exp(-0.5):.4f} (the nu_1 of Basso et al.)")

# ------------------------------------- depth one and Goemans-Williamson on random regular graphs
# For a graph with triangles, edge uv lying in lam_uv triangles, depth-one QAOA gives the expected cut fraction
# (Wang et al. 2018, in the conventions of Lecture 5, checked below by simulation)
#   1/2 - (1/4) sin4b sin g (cos^{d_u} g + cos^{d_v} g) - (1/4) sin^2 2b cos^{d_u + d_v - 2 lam} g (1 - cos^lam 2g),
# with d_u = deg u - 1. On each instance the angles are optimised with this formula; the distribution of the cut
# of one measured string is then computed from the statevector. Goemans-Williamson: the SDP by cvxpy/SCS, as in
# Lecture 1, and the cut of one random hyperplane, from 10^4 hyperplanes. For both methods the mean and the
# standard deviation are those of the cut fraction of a single run on a random instance, over the instances and
# the randomness of the method (the law of total variance).
import networkx as nx, cvxpy as cp
from scipy.optimize import minimize

def depth_one_formula(G, beta, gamma):
    tot = 0.0
    for u, v in G.edges():
        du, dv, lam = G.degree(u) - 1, G.degree(v) - 1, len(set(G[u]) & set(G[v]))
        tot += (0.5 - 0.25 * np.sin(4 * beta) * np.sin(gamma) * (np.cos(gamma) ** du + np.cos(gamma) ** dv)
                - 0.25 * np.sin(2 * beta) ** 2 * np.cos(gamma) ** (du + dv - 2 * lam) * (1 - np.cos(2 * gamma) ** lam))
    return tot

err = 0.0
for seed, (nn, D) in enumerate([(8, 3), (10, 5), (12, 6), (12, 7)]):
    G = nx.random_regular_graph(D, nn, seed=seed)
    hc = cut_diagonal(nn, list(G.edges()))
    for beta, gamma in rng.uniform(-np.pi, np.pi, size=(10, 2)):
        err = max(err, abs(-qaoa_one(nn, hc, beta, gamma) - depth_one_formula(G, beta, gamma)))
print(f"depth one with triangles: max deviation of the formula from simulation {err:.1e}")

NI, INST, DS = 22, 20, list(range(2, 12))
idx = np.arange(2 ** NI, dtype=np.uint32)
bitq = [((idx >> (NI - 1 - q)) & 1).astype(np.int8) for q in range(NI)]
grid = [(b, g) for b in np.linspace(0, np.pi / 2, 31)[:-1] for g in np.linspace(-np.pi, np.pi, 121)[:-1]]
rng_gw = np.random.default_rng(6)
rows = []
for D in DS:
    q_mu, q_var, g_mu, g_var, opt = [], [], [], [], []
    for s in range(INST):
        G = nx.random_regular_graph(D, NI, seed=1000 * D + s)
        E = np.array(G.edges()); m = len(E)
        lam = np.array([len(set(G[u]) & set(G[v])) for u, v in E])
        frac = lambda x: (0.5 - 0.5 * np.sin(4 * x[0]) * np.sin(x[1]) * np.cos(x[1]) ** (D - 1)
                          - 0.25 * np.sin(2 * x[0]) ** 2 * np.cos(x[1]) ** (2 * D - 2 - 2 * lam) * (1 - np.cos(2 * x[1]) ** lam)).mean()
        x0 = max(grid, key=frac)
        beta, gamma = minimize(lambda x: -frac(x), x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-13)).x
        cut = np.zeros(2 ** NI, np.int16)
        for u, v in E:
            cut += bitq[u] ^ bitq[v]
        psi = np.exp(1j * gamma * cut) * 2 ** (-NI / 2)            # e^{-i gamma H_C}|+>, H_C = -cut
        for j in range(NI):                                         # e^{-i beta X_j}
            psi = psi.reshape(2 ** j, 2, 2 ** (NI - 1 - j))
            a, b = psi[:, 0, :].copy(), psi[:, 1, :].copy()
            psi[:, 0, :] = np.cos(beta) * a - 1j * np.sin(beta) * b
            psi[:, 1, :] = np.cos(beta) * b - 1j * np.sin(beta) * a
            psi = psi.reshape(-1)
        p, c = np.abs(psi) ** 2, cut / m
        q_mu.append(p @ c); q_var.append(p @ c ** 2 - (p @ c) ** 2)
        assert abs(q_mu[-1] - frac((beta, gamma))) < 1e-9
        opt.append(c.max())
        L = nx.laplacian_matrix(G, nodelist=range(NI)).toarray().astype(float)
        X = cp.Variable((NI, NI), PSD=True)
        cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), [cp.diag(X) == 1]).solve(solver=cp.SCS, eps=1e-8)
        w, U = np.linalg.eigh((X.value + X.value.T) / 2)
        V = U * np.sqrt(np.clip(w, 0, None))                       # rows: the vectors v_i
        z = np.sign(rng_gw.normal(size=(10_000, NI)) @ V.T); z[z == 0] = 1
        gc = np.mean(z[:, E[:, 0]] != z[:, E[:, 1]], axis=1)
        g_mu.append(gc.mean()); g_var.append(gc.var())
    q_mu, g_mu = np.array(q_mu), np.array(g_mu)
    qs, gs = np.sqrt(np.mean(q_var) + q_mu.var()), np.sqrt(np.mean(g_var) + g_mu.var())
    rows.append((D, q_mu.mean(), qs, g_mu.mean(), gs, np.mean(opt), np.std(opt)))
    print(f"D = {D:2d}, n = {NI}: QAOA {q_mu.mean():.4f} +- {qs:.4f} (instances alone {q_mu.std():.4f}),"
          f" GW {g_mu.mean():.4f} +- {gs:.4f} (instances alone {g_mu.std():.4f}), maximum cut {np.mean(opt):.4f}")
rows = np.array(rows)
np.savetxt(f"{OUT}/depth_one_instances.dat", rows, fmt="%.6f", header="D qaoa qaoasd gw gwsd opt optsd", comments="")
band = np.vstack([np.c_[rows[:, 0], rows[:, 1] + rows[:, 2], rows[:, 3] + rows[:, 4]],      # upper edges, then
                  np.c_[rows[::-1, 0], rows[::-1, 1] - rows[::-1, 2], rows[::-1, 3] - rows[::-1, 4]]])   # lower edges back
np.savetxt(f"{OUT}/depth_one_bands.dat", band, fmt="%.6f", header="D qaoa gw", comments="")

# ------------------------------------------------------- the spectrum along the path
edges = [tuple(map(int, l.split())) for l in open("figures/ch04/maxcut_graph.dat").read().split("\n")[1:] if l.strip()]
n = 8
H0, H1 = -x_sum(n), np.diag(cut_diagonal(n, edges))
optima = np.flatnonzero(np.diag(H1) == np.diag(H1).min())
print(f"Lecture 4 instance: {len(edges)} edges, minimum of H_C {np.diag(H1).min():.0f}, optimal strings "
      + ", ".join(format(x, f"0{n}b") for x in optima))
rows = []
for s in np.linspace(0, 1, 401):
    w = eigh((1 - s) * H0 + s * H1, eigvals_only=True, subset_by_index=[0, 3])
    rows.append((s, *w))
rows = np.array(rows)
np.savetxt(f"{OUT}/crossing_spectrum.dat", rows, fmt="%.6f", header="s l0 l1 l2 l3", comments="")
gap = rows[:, 2] - rows[:, 1]
for s0 in (0.5, 0.9, 0.99, 1.0):
    k = int(round(s0 * 400))
    print(f"s = {rows[k, 0]:.3f}: gap l1 - l0 = {gap[k]:.2e}")

# ------------------------------------------------- the proof in numbers: schedules versus depth
# dist(psi, S_opt) = sqrt(1 - sum over optimal strings of |psi_x|^2), for the state of QAOA at the
# angles of the midpoint schedule of Lecture 5 (gamma_k = s_k dt, beta_k = (1 - s_k) dt, H_0 = -H_M),
# and for the continuous evolution along H(s) = (1-s) H_0 + s H_C.
def dist_opt(psi, opt): return float(np.sqrt(max(0.0, 1 - np.sum(np.abs(psi[opt]) ** 2))))

def continuous(H0, H1, psi0, T, steps=4000):
    psi = psi0.astype(complex)
    for k in range(steps):
        s = (k + 0.5) / steps
        w, U = eigh((1 - s) * H0 + s * H1)
        psi = U @ (np.exp(-1j * (T / steps) * w) * (U.conj().T @ psi))
    return psi

def schedule(mixers, h1, psi0, T, L):
    """mixers: list of (eigenvalues, eigenvectors) of the terms H_i of H_0 = sum_i H_i, applied in order."""
    psi = psi0.astype(complex); dt = T / L
    for k in range(1, L + 1):
        s = (k - 0.5) / L
        psi = np.exp(-1j * s * dt * h1) * psi                  # e^{-i gamma_k H_1}, H_1 diagonal
        for w, U in mixers:                                    # e^{-i beta_k H_0}, or its sequential split
            psi = U @ (np.exp(-1j * (1 - s) * dt * w) * (U.conj().T @ psi))
    return psi

LS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
# MaxCut, transverse field, the instance of Lecture 4 (H0, H1, optima as above)
psi0 = np.full(2 ** n, 2 ** (-n / 2))
w0, U0 = eigh(H0)
TS = [5, 10, 20, 40]
rows, adia = [], []
for T in TS:
    adia.append((T, dist_opt(continuous(H0, H1, psi0, T), optima)))
for L in LS:
    rows.append((L, *[dist_opt(schedule([(w0, U0)], np.diag(H1), psi0, T, L), optima) for T in TS]))
np.savetxt(f"{OUT}/proof_maxcut.dat", rows, fmt="%.6e", header="L " + " ".join(f"T{T}" for T in TS), comments="")
np.savetxt(f"{OUT}/proof_maxcut_adiabatic.dat", adia, fmt="%.6e", header="T dist", comments="")
print("MaxCut, adiabatic distances:", {T: f"{d:.2e}" for T, d in adia})
for r in rows:
    if r[0] in (1, 4, 16, 64, 128, 256): print("   L =", r[0], " ".join(f"T={T}: {d:.2e}" for T, d in zip(TS, r[1:])))

# Maximum bisection of the triangular prism (n = 6, three vertices on each side), ring XY mixer
n6 = 6
prism = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
feas = [x for x in range(2 ** n6) if bin(x).count("1") == 3]
z6 = np.array([[(x >> (n6 - 1 - q)) & 1 for q in range(n6)] for x in feas])
cut6 = np.array([sum(b[u] != b[v] for u, v in prism) for b in z6], float)
h1b = -cut6
opt6 = np.flatnonzero(cut6 == cut6.max())
index = {x: i for i, x in enumerate(feas)}
def xy_term(j, k):
    """(X_j X_k + Y_j Y_k)/2 restricted to the strings with three ones: swaps 01 and 10 on (j, k)."""
    A = np.zeros((len(feas), len(feas)))
    for x in feas:
        bj, bk = (x >> (n6 - 1 - j)) & 1, (x >> (n6 - 1 - k)) & 1
        if bj != bk:
            A[index[x ^ (1 << (n6 - 1 - j)) ^ (1 << (n6 - 1 - k))], index[x]] = 1.0
    return A
ring = [(j, (j + 1) % n6) for j in range(n6)]
HM = sum(xy_term(*e) for e in ring)
wM, UM = eigh(HM)
psi0b = UM[:, -1] * np.sign(UM[:, -1].sum())                   # the Perron vector of H_M on S
assert np.all(psi0b > 0)
even = -sum(xy_term(*e) for e in ring[0::2]); odd = -sum(xy_term(*e) for e in ring[1::2])
sim = [eigh(-HM)]
seq = [eigh(even), eigh(odd)]
TB = [5, 20]
adb = [(T, dist_opt(continuous(-HM, np.diag(h1b), psi0b, T), opt6)) for T in TB]
rows = []
for L in LS:
    r = [L]
    for T in TB:
        r += [dist_opt(schedule(sim, h1b, psi0b, T, L), opt6), dist_opt(schedule(seq, h1b, psi0b, T, L), opt6)]
    rows.append(r)
np.savetxt(f"{OUT}/proof_bisection.dat", rows, fmt="%.6e", header="L " + " ".join(f"sim{T} seq{T}" for T in TB), comments="")
np.savetxt(f"{OUT}/proof_bisection_adiabatic.dat", adb, fmt="%.6e", header="T dist", comments="")
print(f"bisection of the prism: {len(feas)} feasible strings, maximum bisection {cut6.max():.0f}, {len(opt6)} optimal strings;"
      f" Perron vector uniform: {np.allclose(psi0b, psi0b[0])}")
print("bisection, adiabatic distances:", {T: f"{d:.2e}" for T, d in adb})
for r in rows:
    if r[0] in (1, 4, 16, 64, 128, 256): print("   L =", r[0], " ".join(f"{d:.2e}" for d in r[1:]))

# ------------------------------------------------------- the action of the mixer on four qubits
# Bit strings x_1 x_2 x_3 x_4, with x_1 the leading bit; the ring XY mixer and the 4-cycle use the ring 1-2-3-4-1.
n4, b4, g4 = 4, np.pi / 8, -np.pi / 4
bits4 = np.array([[(x >> (n4 - 1 - q)) & 1 for q in range(n4)] for x in range(2 ** n4)])
X1, Y1 = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])
def on(ops):
    """The tensor product of ops[q] on the qubits q in ops and of the identity on the others."""
    M = np.eye(1)
    for q in range(n4):
        M = np.kron(M, ops.get(q, np.eye(2)))
    return M
ring4 = [(j, (j + 1) % n4) for j in range(n4)]
HX4 = sum(on({q: X1}) for q in range(n4))
HXY4 = sum(0.5 * (on({j: X1, k: X1}) + on({j: Y1, k: Y1})) for j, k in ring4)
UX4, UXY4 = expm(-1j * b4 * HX4), expm(-1j * b4 * HXY4)
cut4 = np.array([sum(b[j] != b[k] for j, k in ring4) for b in bits4])
e = np.eye(2 ** n4)
phased = np.exp(-1j * g4 * (-cut4)) / 4                         # e^{-i gamma H_C}|+>^4, with H_C = -cut
states = [e[0], UX4 @ e[0], e[3], UXY4 @ e[3], phased, UX4 @ phased]
probs = [np.abs(v) ** 2 for v in states]
assert np.allclose(probs[3][bits4.sum(axis=1) != 2], 0)        # the XY mixer preserves the number of ones
assert np.allclose(UX4 @ np.full(2 ** n4, 0.25), np.exp(-1j * b4 * n4) * 0.25)   # |+>^4 is an eigenvector
assert abs(probs[5] @ cut4 - 3) < 1e-12                        # the expected cut -F_1 = 2 - sin 4b sin 2g = 3
rows = np.column_stack([np.arange(2 ** n4), *probs, cut4, np.degrees(np.angle(phased))])
np.savetxt(f"{OUT}/mixer_action.dat", rows, fmt=["%d"] + ["%.6f"] * 6 + ["%d", "%.1f"],
           header="i a0 a1 b0 b1 c0 c1 cut ph", comments="")
print("mixer, transverse field from 0000:", np.round(probs[1][[0, 1, 3, 7, 15]], 4), "by Hamming distance 0..4")
print("mixer, ring XY from 0011:", {format(x, "04b"): round(probs[3][x], 4) for x in np.flatnonzero(probs[3] > 1e-12)})
print("mixer, 4-cycle: probabilities after the mixer by cut", {c: round(probs[5][cut4 == c].sum(), 4) for c in (0, 2, 4)})

# ------------------------------------------------- the interpolation heuristic on the instance of Lecture 4
# Zhou et al.: the optimum at depth p, interpolated linearly to p + 1 points, starts BFGS at depth p + 1;
# depth one starts from the best point of a grid on the branch 0 <= beta <= pi/4, -pi/2 <= gamma <= 0.
from scipy.optimize import minimize
lec4 = [tuple(map(int, l.split())) for l in open("figures/ch04/maxcut_graph.dat").read().split("\n")[1:] if l.strip()]
hc8 = cut_diagonal(8, lec4)
mc8 = -hc8.min()
def qaoa_state(n, hc, beta, gamma):
    """e^{-i beta_p H_M} e^{-i gamma_p H_C} ... e^{-i beta_1 H_M} e^{-i gamma_1 H_C} |+>^n."""
    psi = np.full(2 ** n, 2 ** (-n / 2), complex)
    for b, g in zip(beta, gamma):
        psi = psi * np.exp(-1j * g * hc)
        for j in range(n):
            psi = psi.reshape(2 ** (n - 1 - j), 2, 2 ** j)
            a, c = psi[:, 0, :].copy(), psi[:, 1, :].copy()
            psi[:, 0, :] = np.cos(b) * a - 1j * np.sin(b) * c
            psi[:, 1, :] = np.cos(b) * c - 1j * np.sin(b) * a
            psi = psi.reshape(-1)
    return psi
def F8(x):
    p = len(x) // 2
    psi = qaoa_state(8, hc8, x[:p], x[p:])
    return float(np.real(np.vdot(psi, hc8 * psi)))
def local(x0):
    r = minimize(F8, x0, method="BFGS", options=dict(gtol=1e-8))
    return r.x, r.fun
def interpolate(v):
    """The start at depth p + 1 from the angles v at depth p (Zhou et al.), with v_0 = v_{p+1} = 0."""
    p = len(v); w = np.concatenate([[0], v, [0]])
    return np.array([(i - 1) / p * w[i - 1] + (p - i + 1) / p * w[i] for i in range(1, p + 2)])
def interp_run(x1, depth):
    run = {1: local(x1)}
    for p in range(1, depth):
        x = run[p][0]
        run[p + 1] = local(np.concatenate([interpolate(x[:p]), interpolate(x[p:])]))
    return run
grid = [(F8(np.array([b, g])), b, g) for b in np.linspace(0, np.pi / 4, 41) for g in np.linspace(-np.pi / 2, 0, 81)]
_, b1, g1 = min(grid)
PMAX = 10
run = interp_run(np.array([b1, g1]), PMAX)
other = interp_run(np.array([b1 + np.pi / 2, np.pi - g1]), 7)      # an equivalent depth-one optimum, another branch
assert abs(other[1][1] - run[1][1]) < 1e-9
rng_r = np.random.default_rng(7)
rows = []
for p in range(1, PMAX + 1):
    rand = -np.array([local(np.concatenate([rng_r.uniform(-np.pi / 2, np.pi / 2, p), rng_r.uniform(-np.pi, np.pi, p)]))[1]
                      for _ in range(20)]) / mc8
    rows.append((p, 1 + run[p][1] / mc8, 1 - np.median(rand), 1 - rand.max()))
np.savetxt(f"{OUT}/interp_ratio.dat", rows, fmt=["%d", "%.6e", "%.6e", "%.6e"], header="p interp rmed rbest", comments="")
cols, names = [], []
for p in range(2, PMAX + 1, 2):
    x = run[p][0]
    pad = lambda v: np.concatenate([v, np.full(PMAX - p, np.nan)])
    cols += [pad(np.arange(p) / (p - 1)), pad(x[:p]), pad(-x[p:])]
    names += [f"x{p}", f"b{p}", f"g{p}"]
np.savetxt(f"{OUT}/interp_angles.dat", np.column_stack(cols), fmt="%.6f", header=" ".join(names), comments="")
print("interpolation heuristic, depth-one start (beta, gamma) =", (round(run[1][0][0], 4), round(run[1][0][1], 4)))
for r in rows:
    print(f"   p = {r[0]:2d}: ratio {1 - r[1]:.4f}; 20 random starts: median {1 - r[2]:.4f}, best {1 - r[3]:.4f}")
print("from the equivalent start on another branch:", {p: round(-f / mc8, 4) for p, (x, f) in other.items()})
