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
# Q = np.eye(n)
# Build SPD Hessian with given eigenvalues
A = Q @ np.diag(lambdas) @ Q.T

# Random initial error
x0 = rng.normal(size=n)
x_star = rng.uniform(size=n)
e0 = x0 - x_star


# ----------------------------
# AutoSGM implementation
# ----------------------------
def run_autosgm_full(e0, A, beta=0.25, mode=2, iters=20):
    n = len(e0)
    e = e0.copy()
    de = np.zeros_like(e)
    e_hist = []
    lam_max = np.linalg.eigvalsh(A).max()
    lam_min = np.linalg.eigvalsh(A).min()
    print(lam_max)
    beta = 0.8
    if mode == 0:
        # "raw opt" style
        gamma, eta = 0, 1
        Alphas = 2 / (np.sqrt(3) * (lam_max+lam_min))
    elif mode == 1:
        # "raw match"
        gamma, eta = 0, 1
        Alphas = 2 / (lam_max+lam_min)
    elif mode == 2:
        # AutoSGM optimal parameters
        kappa = lam_max/lam_min
        # beta = (np.sqrt((3*kappa)+1)-2)/(np.sqrt((3*kappa)+1)+2)
        # beta = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)
        # gamma = beta / (1 + beta)
        beta = ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**2
        gamma = 0
       
        eta = (1 - beta) / (1 - gamma)

        # alpha = (1 + beta) / (eta * (lam_max+lam_min))
        # alpha = 4 / ((3*lam_max)+lam_min)
        alpha = 4 /( (np.sqrt(lam_max)+np.sqrt(lam_min))**2)
        # alpha = 1/(lam_max)
        
        Alphas = alpha
    else:
        raise ValueError("Unknown mode")

    for k in range(iters):
        Bp = beta * np.eye(n) - eta * Alphas * A
        Kp = - eta * (1 - gamma) * Alphas * A

        deprev = de.copy()
        de = Bp @ de + Kp @ e
        e = e + deprev

        e_hist.append(np.linalg.norm(e))
        if np.linalg.norm(e) < 1e-15:
            break

    return np.array(e_hist)


# ----------------------------
# Conjugate Gradient
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
        beta_k = (r_new @ r_new) / (r @ r)
        p = r_new + beta_k * p
        r = r_new
    return np.array(e_hist)


# ----------------------------
# Run simulations
# ----------------------------
iters = 20
autosgm_err0 = run_autosgm_full(e0, A, beta=0.25, mode=0, iters=iters)
autosgm_err1 = run_autosgm_full(e0, A, beta=0.25, mode=1, iters=iters)
autosgm_err2 = run_autosgm_full(e0, A, beta=0.25, mode=2, iters=iters)
cg_err = run_cg(A, e0, iters=iters)

# ----------------------------
# Plot results
# ----------------------------
plt.semilogy(autosgm_err0, label="AutoSGM raw opt")
plt.semilogy(autosgm_err1, label="AutoSGM raw match")
plt.semilogy(autosgm_err2, label="AutoSGM opt")
plt.semilogy(cg_err, label="Conjugate Gradient", lw=1, ls='dashed')
plt.xlabel("Iteration")
plt.ylabel("Error norm (log scale)")
plt.legend()
plt.title("Convergence comparison: AutoSGM vs CG (non-diagonal Hessian)")
plt.show()
