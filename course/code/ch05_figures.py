"""Numerical data for the figures of Lecture 5 (the variational quantum eigensolver, and the
hardness of training after Bittel and Kliesch).

Writes, into figures/ch05/:
  local_minima.dat   histogram of the cut ratio reached on a random 3-regular graph on 20 vertices
                     by gradient descent on continuous MaxCut mu(phi), and by greedy single-vertex
                     local search, each from 1000 random starts (bin centre, fractions)
  onelayer_curve.dat mu(E t) along the single-layer curve for the path on 3 vertices with the clock
                     energies E_j = 2 pi / 6^j, 0 <= t < 6^3
"""
import itertools, os
import numpy as np

OUT = "figures/ch05"
os.makedirs(OUT, exist_ok=True)

def random_cubic(n, rng):
    """Random simple 3-regular graph on n vertices (pairing model with rejection)."""
    while True:
        stubs = np.repeat(np.arange(n), 3)
        rng.shuffle(stubs)
        edges = {tuple(sorted(stubs[i:i + 2])) for i in range(0, 3 * n, 2)}
        if len(edges) == 3 * n // 2 and all(u != v for u, v in edges):
            return sorted(edges)

def adjacency(n, edges):
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    return A

def cut(A, v):
    """Cut value of the +-1 vectors in the rows of v."""
    return 0.25 * (A.sum() - np.einsum("si,ij,sj->s", v, A, v))

# --------------------------------------------------------------------------- local minima
n, starts = 20, 1000
rng = np.random.default_rng(5)
edges = random_cubic(n, rng)
A = adjacency(n, edges)
z = 1 - 2 * np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)
mc = max(cut(A, z[i:i + 2**16].astype(float)).max() for i in range(0, len(z), 2**16))

# gradient descent on mu(phi) = 1/4 sum_ij A_ij (cos phi_i cos phi_j - 1)
phi = rng.uniform(0, 2 * np.pi, (starts, n))
for _ in range(20000):
    grad = -0.5 * np.sin(phi) * (np.cos(phi) @ A)
    phi -= 0.1 * grad
    if np.abs(grad).max() < 1e-8:
        break
gd = cut(A, np.sign(np.cos(phi))) / mc

# greedy single-vertex local search from random cuts
greedy = []
for _ in range(starts):
    v = rng.choice([-1.0, 1.0], n)
    while True:
        gain = v * (A @ v)            # flipping i changes the cut by v_i (A v)_i
        i = int(np.argmax(gain))
        if gain[i] <= 0:
            break
        v[i] = -v[i]
    greedy.append(cut(A, v[None, :])[0] / mc)
greedy = np.array(greedy)

with open(f"{OUT}/local_minima.dat", "w") as f:
    f.write("ratio gd greedy\n")
    for c in np.arange(int(0.7 * mc), mc + 1) / mc:
        f.write(f"{c:.6f} {np.mean(np.isclose(gd, c)):.6f} {np.mean(np.isclose(greedy, c)):.6f}\n")
print(f"cubic graph n={n}, |E|={len(edges)}, MaxCut={mc:.0f}, |E|/2={len(edges)/2}")
print(f"gradient descent: optimum reached {np.mean(np.isclose(gd, 1)):.3f}, worst ratio {gd.min():.3f}, mean {gd.mean():.3f}")
print(f"greedy search:    optimum reached {np.mean(np.isclose(greedy, 1)):.3f}, worst ratio {greedy.min():.3f}, mean {greedy.mean():.3f}")

# --------------------------------------------------------------------------- one layer
m, d = 6, 3
P3 = adjacency(3, [(0, 1), (1, 2)])
E = 2 * np.pi / m ** np.arange(1, d + 1)
def mu_curve(t):
    c = np.cos(np.outer(t, E))
    return 0.25 * (np.einsum("si,ij,sj->s", c, P3, c) - P3.sum())
t = np.linspace(0, m**d, 3001)
with open(f"{OUT}/onelayer_curve.dat", "w") as f:
    f.write("t mu\n")
    for ti, mi in zip(t, mu_curve(t)):
        f.write(f"{ti:.4f} {mi:.6f}\n")
tf = np.linspace(0, m**d, 2_000_001)
mf = mu_curve(tf)
interior = (mf[1:-1] < mf[:-2]) & (mf[1:-1] < mf[2:])
k = np.argmin(mf)
print(f"one layer, P3, m={m}: {interior.sum()} local minima on [0, {m**d}), best mu {mf[k]:.4f} at t = {tf[k]:.3f}")
