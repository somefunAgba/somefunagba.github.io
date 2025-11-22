import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Problem setup
# ----------------------------
n = 10
rng = np.random.default_rng(42)

# Spectrum: from 1 to 1e4 to stress conditioning
lambdas = np.logspace(0, 4, n)

# Random orthogonal basis
Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
# Build SPD Hessian with given eigenvalues
H = Q @ np.diag(lambdas) @ Q.T

# Random initial error
x0 = rng.normal(size=n)
x_star = rng.uniform(size=n)
e0 = x0 - x_star

# ----------------------------
# AutoSGM with Rayleigh α_t
# ----------------------------
def run_autosgm_rayleigh(e0, H, beta=0.25, iters=20):
    n = len(e0)
    e = e0.copy()
    de = np.zeros_like(e)
    e_hist = []

    # Fixed gamma from derivation
    gamma = 0
    eta = (1 - beta) / (1 - gamma)

    I = np.eye(n)

    for t in range(iters):
        # Rayleigh quotient curvature estimate
        lam_hat = (e.T @ (H @ e)) / (e.T @ e)
        alpha_t = (1 + beta) / (eta * lam_hat)

        # System matrices
        Bp = beta * I - alpha_t * eta * H
        Kp = - alpha_t * eta * (1 - gamma) * H

        # Update states
        deprev = de.copy()
        de = Bp @ de + Kp @ e
        e = e + deprev

        e_hist.append(np.linalg.norm(e))
        if e_hist[-1] < 1e-15:
            break

    return np.array(e_hist)

# ----------------------------
# Conjugate Gradient for comparison
# ----------------------------
def run_cg(A, e0, iters=20):
    x = e0.copy()
    r = -A @ x
    p = r.copy()
    e_hist = [np.linalg.norm(x)]
    for k in range(iters):
        Ap = A @ p
        alpha_k = (r @ r) / (p @ Ap)
        x = x + alpha_k * p
        r_new = r - alpha_k * Ap
        e_hist.append(np.linalg.norm(x))
        if np.linalg.norm(r_new) < 1e-15:
            break
        beta_k = 0 #(r_new @ r_new) / (r @ r)
        p = r_new + beta_k * p
        r = r_new
    return np.array(e_hist)

# ----------------------------
# Run simulations
# ----------------------------
iters = 20
autosgm_err_rayleigh = run_autosgm_rayleigh(e0, H, beta=0, iters=iters)
cg_err = run_cg(H, e0, iters=iters)

# ----------------------------
# Plot results
# ----------------------------
plt.semilogy(autosgm_err_rayleigh, label="AutoSGM (Rayleigh α_t)")
plt.semilogy(cg_err, label="Conjugate Gradient", lw=1, ls='dashed')
plt.xlabel("Iteration")
plt.ylabel("Error norm (log scale)")
plt.legend()
plt.title("Convergence comparison: AutoSGM (Rayleigh α_t) vs CG")
plt.show()
