"""Classical optimizers on the worked example of Lecture 5, for Lecture 6.

Depth-one QAOA for MaxCut on the 4-cycle, F(beta, gamma) = -2 + sin(4 beta) sin(2 gamma), in the
conventions of Lecture 5: energies are estimated from measured copies, and a gradient from the
parameter-shift rule gate by gate, 16 circuits (2 for each of the 4 R_ZZ and 4 R_X gates).
All four methods start at (0.1, -0.1), as in Lecture 5, with a budget of 600 000 measured copies,
that is, 600 circuits of M = 1000 copies.

  COBYLA   SciPy, initial trust-region radius 0.2, M = 1000 copies per circuit
  SPSA     gains a_k = a/(k+1+A)^0.602, c_k = c/(k+1)^0.101 (Spall), a = 0.2, A = 10, c = 0.1; M = 1000
  BFGS     SciPy, with parameter-shift gradients; M = 1000
  STORM    the stochastic trust-region method of Chen, Menickelly and Scheinberg, as analysed by
           Blanchet, Cartis, Menickelly and Scheinberg: a linear model from a parameter-shift gradient
           with error of order delta, values with error of order delta^2, so that the number of copies
           grows as the trust region shrinks; step to the boundary of the trust region along -g.

For COBYLA the current point is the point with the best estimated energy so far, which is its iterate;
for the other methods it is the iterate. Writes, into figures/ch06/, opt_<method>.dat with the columns
copies xplot beta gamma F excess (the exact energy at the current point, and its excess over the
minimum -3; xplot is the number of copies with the start drawn at 10^3), and prints a summary.
"""
import os, warnings
import numpy as np
from scipy.optimize import minimize

OUT = "figures/ch06"
os.makedirs(OUT, exist_ok=True)
n, edges, M, BUDGET = 4, [(0, 1), (1, 2), (2, 3), (3, 0)], 1000, 600_000
bits = np.array([[(x >> (n - 1 - q)) & 1 for q in range(n)] for x in range(2 ** n)])
cuts = np.array([sum(b[i] != b[j] for i, j in edges) for b in bits])
hc = -cuts.astype(float)

def state(gz, gx):
    """e^{-i sum_j gx_j X_j / 2} e^{-i sum_e gz_e Z_iZ_j / 2} |+>^n, angles per gate (R_ZZ(gz_e), R_X(gx_j))."""
    z = 1 - 2 * bits
    phase = sum(gz[e] * z[:, i] * z[:, j] for e, (i, j) in enumerate(edges))
    psi = np.full(2 ** n, 2 ** (-n / 2), complex) * np.exp(-0.5j * phase)
    for q in range(n):
        psi = psi.reshape(2 ** q, 2, 2 ** (n - 1 - q))
        a, b = psi[:, 0, :].copy(), psi[:, 1, :].copy()
        c, s = np.cos(gx[q] / 2), np.sin(gx[q] / 2)
        psi[:, 0, :], psi[:, 1, :] = c * a - 1j * s * b, c * b - 1j * s * a
        psi = psi.reshape(-1)
    return psi

def angles(beta, gamma): return np.full(len(edges), gamma), np.full(n, 2 * beta)
def F_exact(beta, gamma): return -2 + np.sin(4 * beta) * np.sin(2 * gamma)
assert abs(float(np.real(np.vdot(state(*angles(0.3, -0.5)), hc * state(*angles(0.3, -0.5))))) - F_exact(0.3, -0.5)) < 1e-12

class Counter:
    """Estimates energies from measured copies, counts copies and records the current point."""
    def __init__(self, rng, mode):
        self.rng, self.mode, self.copies, self.rows, self.best = rng, mode, 0, [], (np.inf, None)
    def shots(self, gz, gx, m=M):
        self.copies += m
        p = np.abs(state(gz, gx)) ** 2
        return float(np.mean(hc[self.rng.choice(2 ** n, size=m, p=p / p.sum())]))
    def F(self, x, m=M):
        v = self.shots(*angles(*x), m)
        if self.mode == "best":
            if v < self.best[0]:
                self.best = (v, np.array(x, float))
            self.record(self.best[1])
        return v
    def grad(self, x, m=M):
        gz, gx = angles(*x)
        g = np.zeros(2)
        for e in range(len(edges)):
            p, q_ = gz.copy(), gz.copy(); p[e] += np.pi / 2; q_[e] -= np.pi / 2
            g[1] += 0.5 * (self.shots(p, gx, m) - self.shots(q_, gx, m))
        for q in range(n):
            p, q_ = gx.copy(), gx.copy(); p[q] += np.pi / 2; q_[q] -= np.pi / 2
            g[0] += 2 * 0.5 * (self.shots(gz, p, m) - self.shots(gz, q_, m))
        return g
    def record(self, x):
        self.rows.append((self.copies, x[0], x[1], F_exact(*x)))

x0 = np.array([0.1, -0.1])
results = {}

# COBYLA: values only
c = Counter(np.random.default_rng(6), "best")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    minimize(c.F, x0, method="COBYLA", options=dict(rhobeg=0.2, maxiter=BUDGET // M, tol=1e-6))
results["cobyla"] = c

# SPSA: two evaluations per iteration
c = Counter(np.random.default_rng(6), "iterate")
x, a, A, cc = x0.copy(), 0.2, 10, 0.1
c.record(x)
k = 0
while c.copies + 2 * M <= BUDGET:
    ak, ck = a / (k + 1 + A) ** 0.602, cc / (k + 1) ** 0.101
    d = c.rng.choice([-1.0, 1.0], size=2)
    g = (c.F(x + ck * d) - c.F(x - ck * d)) / (2 * ck) * d
    x = x - ak * g
    c.record(x); k += 1
results["spsa"] = c

# BFGS with parameter-shift gradients
c = Counter(np.random.default_rng(6), "iterate")
c.record(x0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = minimize(c.F, x0, jac=c.grad, method="BFGS", callback=c.record, options=dict(maxiter=60, gtol=1e-8))
results["bfgs"] = c
print(f"BFGS stopped: {res.message} after {res.nit} iterations, {c.copies} copies")

# STORM with linear models: values to accuracy kappa_f delta^2, gradients to accuracy kappa_g delta
c = Counter(np.random.default_rng(6), "iterate")
SIG_F, SIG_G, KF, KG = 1.0, 3.0, 1.0, 1.0           # per-copy standard deviations of F and of a gradient component
eta1, eta2, gam, delta, dmax = 0.1, 0.5, 2.0, 0.2, 0.5
x = x0.copy(); c.record(x); log = []
while True:
    mf = int(np.ceil((SIG_F / (KF * delta ** 2)) ** 2)); mg = int(np.ceil((SIG_G / (KG * delta)) ** 2))
    if c.copies + 16 * mg + 2 * mf > BUDGET:
        break
    g = c.grad(x, mg)
    s = -delta * g / max(np.linalg.norm(g), 1e-12)
    f0, fs = c.F(x, mf), c.F(x + s, mf)
    rho = (f0 - fs) / (delta * np.linalg.norm(g))
    ok = rho >= eta1 and np.linalg.norm(g) >= eta2 * delta
    if ok:
        x = x + s; delta = min(gam * delta, dmax)
    else:
        delta = delta / gam
    log.append((ok, delta, mf, mg))
    c.record(x)
results["storm"] = c
print(f"STORM: {len(log)} iterations, {sum(o for o, *_ in log)} successful; final delta {delta:.4f},"
      f" copies per value {log[-1][2]}, per shifted circuit {log[-1][3]}")

for name, c in results.items():
    rows = [r for r in c.rows if r[0] <= BUDGET]
    rows = [r for i, r in enumerate(rows) if i == 0 or i == len(rows) - 1 or (r[1], r[2]) != (rows[i - 1][1], rows[i - 1][2])]
    with open(f"{OUT}/opt_{name}.dat", "w") as f:
        f.write("copies xplot beta gamma F excess\n")          # xplot: copies, with the start drawn at 10^3
        for r in rows:
            f.write(f"{r[0]} {max(r[0], 1000)} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[3] + 3:.6e}\n")
    F = np.array([r[3] for r in rows]); cop = np.array([r[0] for r in rows])
    hit = cop[np.argmax(F < -2.95)] if np.any(F < -2.95) else None
    print(f"{name:7s}: {len(rows)} points, last at {cop[-1]} copies: (beta, gamma) = ({rows[-1][1]:.3f}, {rows[-1][2]:.3f}),"
          f" F = {F[-1]:.4f}; first F < -2.95 after {hit} copies")
    at = {b: F[cop <= b][-1] for b in (2e4, 5e4, 1e5, 2e5, 6e5) if np.any(cop <= b)}
    print("         F at budgets", {int(b): round(v, 4) for b, v in at.items()})
