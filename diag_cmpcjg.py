import numpy as np

# Problem setup
n = 2
# Spectrum: from 1 to 1e4 to stress conditioning
lambdas = np.logspace(0, 4, n)  # diag(A)
A = np.diag(lambdas)
rng = np.random.default_rng(42)
x0 = rng.normal(size=n)
x_star = rng.uniform(size=n)
e0 = x0 - x_star

# AutoSGM parameters (per-coordinate)
beta = 0.25  # shared momentum pole in (0,1)
# gamma = -(1 - beta)/ beta  # filter zero for two-step property per mode
# alpha = 1.0 / (3.0 * lambdas)  # per-coordinate step sizes

# def autosgm_step(e_prev, e_curr, lambdas, alpha, beta, gamma, m_prev):
#     # per-coordinate gradient at x = x_star + e_curr: g_i = lambda_i * e_i
#     g = lambdas * e_curr
#     # filter (first-order LTI): m_k = beta * m_{k-1} + g - gamma * g_prev
#     # implement with stored previous gradient per coordinate
#     # for simplicity, treat g_prev via state carried in m_prev/approx;
#     # more explicit: keep g_prev separately
#     # Here we keep g_prev separately:
#     return g

# Explicit stateful AutoSGM implementation
def run_autosgm(e0, lambdas, beta=0.2, mode=0, iters=20):
    e = e0.copy()
    # e_hist = [np.linalg.norm(e)]
    e_hist = []
    # print(lambdas)
    # keep filtered state
    de = np.zeros_like(e)

    if mode == 0:
        alphas =  1/(np.sqrt(3) * lambdas) 
        beta, gamma, eta = 0, 0, 1
    if mode == 1:
        alphas =  1/(1 * lambdas) 
        beta, gamma, eta = 0, 0, 1
    if mode == 2:
        gamma = (beta)/(1+beta)     
        # eta = 1/(1-gamma)
        # alphas = (1+beta)/(eta * lambdas) 
        # print('0',alphas)
        eta = (1-beta)/(1-gamma)
        alphas = (1+beta)/(eta * lambdas) 
        # print('1', alphas)
         
    gprev =  0
    for k in range(iters):
        g = lambdas * e
        # state parameters
        bp = beta - alphas*eta*lambdas
        kp = - alphas*eta*lambdas*(1-gamma)
        # change-level
        deprev = 1*de
        de = bp*de + kp*e
        # error, forward
        e = e + deprev        
        e_hist.append(np.linalg.norm(e*e))
        if np.linalg.norm(e*e) < 1e-15:
            break

    return np.array(e_hist)

# Conjugate Gradient (CG) for Ax = b with x* = 0 => b = 0, but we want error dynamics.
# To compare fairly, solve A x = 0 (trivial); instead, simulate CG reducing error in Ax = b form.
# Let b = A x_star = 0; start from x0 and run CG minimizing f(x) = 0.5*(x^T A x).
def run_cg(A, x0, iters=10):
    # Standard CG for SPD A, b=0, minimization of quadratic => r0 = -A x0
    x = x0.copy()
    r = -A @ x  # since b=0
    p = r.copy()
    e_hist = [np.linalg.norm((x - np.zeros_like(x)**2))]
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

# Run simulations
iters = 20
autosgm_err0 = run_autosgm(e0, lambdas, beta, iters=iters, mode=0)
autosgm_err1 = run_autosgm(e0, lambdas, beta, iters=iters, mode=1)
autosgm_err2 = run_autosgm(e0, lambdas, beta, iters=iters, mode=2)
cg_err = run_cg(A, e0, iters=10)

aerrs =  [autosgm_err0, autosgm_err1, autosgm_err2]
# aerrs =  [autosgm_err2, ]
for autosgm_err in aerrs:
    print("AutoSGM error norms:", autosgm_err )
print("CG error norms:", cg_err)


import matplotlib.pyplot as plt

# After running autosgm_err and cg_err from the simulation
plt.semilogy(autosgm_err0, label="AutoSGM raw opt")
plt.semilogy(autosgm_err1, label="AutoSGM raw match)")
plt.semilogy(autosgm_err2, label="AutoSGM opt")
plt.semilogy(cg_err, label="Conjugate Gradient", lw=1, ls='dashed')
plt.xlabel("Iteration")
plt.ylabel("Error norm (log scale)")
plt.legend()
plt.title("Convergence comparison: AutoSGM vs CG (10D diagonal Hessian)")
plt.show()