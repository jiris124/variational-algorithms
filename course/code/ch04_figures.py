"""Numerical data for the figures of Lecture 4 (adiabatic computing and quantum annealing).

Writes, into figures/ch04/:
  maxcut_graph.dat     edges of the 3-regular MaxCut instance (u v)
  maxcut_spectrum.dat  s and the four lowest eigenvalues of -(1-s) sum_j X_j + s H_C
                       in the sector invariant under flipping all spins
  history_ground.dat   ground-state amplitudes of the history-chain Hamiltonian
                       (Aharonov et al.) on L+1 sites, for several s
  walk_quantum_exit.dat    probability of the quantum walk e^{-iAt} being at EXIT (n = 10)
  walk_classical_exit.dat  the same for the classical random walk, after t steps
  walk_columns.dat         distributions over the columns: classical after 200 steps,
                           quantum at the time of its largest probability at EXIT
"""
import itertools, os
import numpy as np

OUT = "figures/ch04"
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- MaxCut
def random_cubic(n, rng):
    """Random simple 3-regular graph on n vertices (pairing model with rejection)."""
    while True:
        stubs = np.repeat(np.arange(n), 3)
        rng.shuffle(stubs)
        edges = {tuple(sorted(stubs[i:i + 2])) for i in range(0, 3 * n, 2)}
        if len(edges) == 3 * n // 2 and all(u != v for u, v in edges):
            return sorted(edges)

def cut_values(n, edges):
    z = np.array(list(itertools.product([0, 1], repeat=n)))
    return sum((z[:, u] != z[:, v]).astype(int) for u, v in edges)

n = 8
rng = np.random.default_rng(7)
while True:
    edges = random_cubic(n, rng)
    cuts = cut_values(n, edges)
    best = cuts.max()
    if best < len(edges) and (cuts == best).sum() == 2:   # non-bipartite, unique up to complement
        break
np.savetxt(f"{OUT}/maxcut_graph.dat", np.array(edges), fmt="%d", header="u v", comments="")
print(f"MaxCut instance: {len(edges)} edges, maximum cut {best}, edges {edges}")

# Even sector: basis (|z> + |~z>)/sqrt 2 with z_0 = 0, since the evolution commutes with prod_j X_j.
dim = 2 ** (n - 1)
reps = [tuple(int(b) for b in format(k, f"0{n}b")) for k in range(2 ** n) if format(k, f"0{n}b")[0] == "0"]
index = {z: i for i, z in enumerate(reps)}
def canon(z):
    return z if z[0] == 0 else tuple(1 - b for b in z)
HC = np.diag([-float(sum(z[u] != z[v] for u, v in edges)) for z in reps])
HX = np.zeros((dim, dim))
for i, z in enumerate(reps):
    for j in range(n):
        w = list(z); w[j] = 1 - w[j]
        HX[index[canon(tuple(w))], i] -= 1.0
rows = []
for s in np.linspace(0, 1, 201):
    ev = np.linalg.eigvalsh((1 - s) * HX + s * HC)
    rows.append([s] + list(ev[:4]))
rows = np.array(rows)
np.savetxt(f"{OUT}/maxcut_spectrum.dat", rows, fmt="%.6f", header="s l0 l1 l2 l3", comments="")
k = np.argmin(rows[:, 2] - rows[:, 1])
print(f"smallest gap {rows[k, 2] - rows[k, 1]:.4f} at s = {rows[k, 0]:.3f}")

# --------------------------------------------------------------------------- history chain
L = 20
def history(s):
    H = np.diag([s / 2] + [1.0] * (L - 1) + [1 - s / 2])
    for l in range(L):
        H[l, l + 1] = H[l + 1, l] = -s / 2
    return H
cols = [np.arange(L + 1)]
for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
    w, v = np.linalg.eigh(history(s))
    a = v[:, 0] * np.sign(v[0, 0])
    assert np.all(np.diff(a) <= 1e-12), "ground state not monotone"
    cols.append(a)
    print(f"history chain s={s:.2f}: gap {w[1] - w[0]:.4f}")
np.savetxt(f"{OUT}/history_ground.dat", np.array(cols).T, fmt="%.6f",
           header="l s0 s025 s05 s075 s1", comments="")

# --------------------------------------------------------------------------- glued trees, n = 10
m = 10
J = 2 * m + 2                                   # columns 0 .. 2m+1
# Classical random walk on the graph, projected to columns (exactly Markov there).
P = np.zeros((J, J))
P[0, 1] = 1.0                                   # ENTRANCE: two children
P[J - 1, J - 2] = 1.0                           # EXIT: two children
for j in range(1, m + 1):                       # left tree incl. its leaves (column m)
    P[j, j - 1], P[j, j + 1] = 1 / 3, 2 / 3
for j in range(m + 1, J - 1):                   # right tree incl. its leaves (column m+1)
    P[j, j - 1], P[j, j + 1] = 2 / 3, 1 / 3
# Quantum walk e^{-iAt} on the column line (unscaled A: sqrt 2, and 2 in the middle).
A = np.zeros((J, J))
for j in range(J - 1):
    A[j, j + 1] = A[j + 1, j] = 2.0 if j == m else np.sqrt(2)
w, V = np.linalg.eigh(A)
def quantum(t):
    psi = V @ (np.exp(-1j * w * t) * V[0, :])
    return np.abs(psi) ** 2
ts = np.linspace(0, 40, 801)
qexit = np.array([quantum(t)[J - 1] for t in ts])
tstar = ts[np.argmax(qexit)]
np.savetxt(f"{OUT}/walk_quantum_exit.dat", np.column_stack([ts, qexit]), fmt="%.6f", header="t quantum", comments="")
# The column walk alternates between even and odd columns, so we average consecutive steps.
pc = np.zeros(J); pc[0] = 1.0
hist = []
for t in range(0, 2002):
    hist.append(pc.copy())
    pc = pc @ P
avg = [(hist[t] + hist[t + 1]) / 2 for t in range(2001)]
crows = [[t, avg[t][J - 1]] for t in range(0, 2001, 5)]
p200 = avg[200]
np.savetxt(f"{OUT}/walk_classical_exit.dat", np.array(crows), fmt="%.6e", header="t classical", comments="")
np.savetxt(f"{OUT}/walk_columns.dat", np.column_stack([np.arange(J), p200, quantum(tstar)]),
           fmt="%.6f", header="j classical quantum", comments="")
print(f"quantum walk: largest probability at EXIT {qexit.max():.3f} at t = {tstar:.2f}; "
      f"classical walk at EXIT after 200 steps {p200[J - 1]:.2e}, after 2000 steps {crows[-1][1]:.2e}, 1/N = {1/(2**(m+2)-2):.2e}")

# --------------------------------------------------------------------------- glued trees, TikZ (n = 4)
def glued_trees_tikz(n=4, seed=3, path=f"{OUT}/gluedtrees.tex"):
    """TikZ picture of two binary trees of depth n glued by a random cycle, with annotations."""
    rng = np.random.default_rng(seed)
    H = 3.0                                          # half-height of the widest column
    cols = 2 * n + 2
    def ypos(j, k):
        size = 2 ** j if j <= n else 2 ** (2 * n + 1 - j)
        return 0.0 if size == 1 else H * (1 - 2 * k / (size - 1))
    L = []
    for j in range(cols):
        size = 2 ** j if j <= n else 2 ** (2 * n + 1 - j)
        for k in range(size):
            L.append(rf"\coordinate (v{j}-{k}) at ({j},{ypos(j, k):.3f});")
    E = []
    for j in range(n):                               # left tree
        for k in range(2 ** j):
            E += [(f"v{j}-{k}", f"v{j+1}-{2*k}"), (f"v{j}-{k}", f"v{j+1}-{2*k+1}")]
    for j in range(n):                               # right tree, root at column 2n+1
        c = 2 * n + 1 - j
        for k in range(2 ** j):
            E += [(f"v{c}-{k}", f"v{c-1}-{2*k}"), (f"v{c}-{k}", f"v{c-1}-{2*k+1}")]
    l, r = rng.permutation(2 ** n), rng.permutation(2 ** n)
    G = []                                           # random cycle l1 r1 l2 r2 ... alternating sides
    for i in range(2 ** n):
        G += [(f"v{n}-{l[i]}", f"v{n+1}-{r[i]}"), (f"v{n+1}-{r[i]}", f"v{n}-{l[(i+1) % 2**n]}")]
    out = [r"\begin{tikzpicture}[x=\gtX,y=\gtY,every node/.style={font=\scriptsize}]"]
    out += L
    out.append(r"\fill[blue!8,rounded corners=4pt] (1.7,-1.35) rectangle (2.3,1.35);")
    out += [rf"\draw[gray!70] ({a}) -- ({b});" for a, b in E]
    out += [rf"\draw[red!60!black,thin] ({a}) -- ({b});" for a, b in G]
    for j in range(cols):
        size = 2 ** j if j <= n else 2 ** (2 * n + 1 - j)
        for k in range(size):
            out.append(rf"\fill[blue!70!black] (v{j}-{k}) circle (1.6pt);")
        out.append(rf"\node[gray] at ({j},{H+0.55}) {{$j{{=}}{j}$}};")
    out += [
        rf"\fill[green!50!black] (v0-0) circle (3pt); \fill[orange!80!black] (v{cols-1}-0) circle (3pt);",
        r"\node[note,anchor=east] (ne) at (-0.35,1.6) {ENTRANCE: one of only\\two vertices of degree $2$;\\we know its name};",
        r"\draw[notearrow] (ne.south) to[bend right=20] (v0-0);",
        rf"\node[note,anchor=west] (nx) at ({cols-1}.35,1.6) {{EXIT: the other one;\\find its name}};",
        rf"\draw[notearrow] (nx.south) to[bend left=20] (v{cols-1}-0);",
        r"\node[note,anchor=north east] (nc) at (2.1,-3.5) {column $j$: $2^j$ vertices;\\$\ket{{\rm col}_j}$ is their uniform\\superposition};",
        r"\draw[notearrow] (nc.north east) -- (2,-1.4);",
        rf"\node[note,anchor=north] (ng) at ({n}.5,-4.7) {{a random cycle glues the\\leaves: each has two\\neighbours across}};",
        rf"\draw[notearrow] (ng.north) -- ({n}.5,-2.2);",
        rf"\node[note,anchor=north west] (nm) at ({n+2.6},-3.5) {{half of all vertices sit\\in the two middle columns:\\a classical walk gets lost}};",
        rf"\draw[notearrow] (nm.north west) to[bend right=15] ({n+1}.05,-2.7);",
        r"\node[note,anchor=south] (nn) at (2.6,4.25) {every vertex has a random $2n$-bit name;\\the oracle only lists the names of its neighbours};",
        r"\draw[notearrow] (nn.south) to[bend right=10] (v3-1);",
        r"\end{tikzpicture}",
    ]
    open(path, "w").write("% Generated by code/ch04_figures.py; do not edit.\n" + "\n".join(out) + "\n")

glued_trees_tikz()
print("wrote figures/ch04/gluedtrees.tex")
