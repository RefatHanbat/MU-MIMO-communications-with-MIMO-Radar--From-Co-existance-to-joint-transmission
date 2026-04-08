
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# ============================================================
# Figure07_reproduce_corrected.py
#
# Corrected rebuild of Figure 7 for:
# Fan Liu et al., "MU-MIMO Communications With MIMO Radar:
# From Co-Existence to Joint Transmission", IEEE TWC, 2018.
#
# Main corrections compared with the previous broken version:
# 1) Weighted total-power manifold solver follows the paper's
#    hypersphere formulation using eqs. (30)-(40), (44)-(51).
# 2) Weighted per-antenna solver uses the paper's X = T^H
#    formulation on the complex oblique manifold using
#    eqs. (52)-(65), instead of reusing the total-power variable.
# 3) Table II weights are used exactly:
#       Sum-Square: total [10,1], per-ant [3,1]
#       Max:        total [10,1], per-ant [1,2]
# 4) Weighted RCG is initialized from the constrained SDR
#    solution at the same Gamma point, which avoids the
#    off-screen wrong branch that caused the earlier plot to
#    show only two visible curves.
#
# Important note:
# Exact point-for-point equality with the paper is still not
# guaranteed because the paper does not publish its exact
# random channel realization. But this script fixes the core
# implementation errors of the previous version.
# ============================================================


def db2lin(x_db):
    return 10.0 ** (x_db / 10.0)


def dBm2W(x_dBm):
    return 10.0 ** ((x_dBm - 30.0) / 10.0)


def cn(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def herm(A):
    return A.conj().T


def steering_vector(N, theta_deg, d=0.5):
    theta = np.deg2rad(theta_deg)
    n = np.arange(N)
    return np.exp(1j * 2.0 * np.pi * d * np.sin(theta) * n)[:, None]


def beampattern(C, angle_grid_deg):
    vals = []
    N = C.shape[0]
    for th in angle_grid_deg:
        a = steering_vector(N, th)
        vals.append(np.real((herm(a) @ C @ a).item()))
    return np.maximum(np.array(vals), 1e-15)


def get_solvers():
    installed = set(cp.installed_solvers())
    out = []
    for s in [cp.MOSEK, cp.CVXOPT, cp.CLARABEL, cp.SCS]:
        try:
            nm = s if isinstance(s, str) else s.name()
        except Exception:
            nm = str(s)
        if nm in installed:
            out.append(s)
    if not out:
        out = [cp.SCS]
    return out


def solve_with_fallback(prob, solvers, verbose=False):
    last_exc = None
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=verbose)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return True, solver, prob.status
        except Exception as e:
            last_exc = e
    return False, None, f"{prob.status}; last_exception={last_exc}"


# ------------------------------------------------------------
# Radar-only 3 dB reference: paper eq. (10)
# ------------------------------------------------------------
def solve_radar_only_shared_3db(N, P0, angle_grid_deg, theta0=0.0,
                                beamwidth_3db=10.0, solvers=None, verbose=False):
    theta1 = theta0 - beamwidth_3db / 2.0
    theta2 = theta0 + beamwidth_3db / 2.0

    R = cp.Variable((N, N), hermitian=True)
    t = cp.Variable()

    a0 = steering_vector(N, theta0).flatten()
    a1 = steering_vector(N, theta1).flatten()
    a2 = steering_vector(N, theta2).flatten()

    p0 = cp.real(cp.quad_form(a0, R))
    p1 = cp.real(cp.quad_form(a1, R))
    p2 = cp.real(cp.quad_form(a2, R))

    cons = [
        R >> 0,
        cp.diag(R) == (P0 / N) * np.ones(N),
        p1 == p0 / 2.0,
        p2 == p0 / 2.0,
    ]

    sidelobe_angles = [th for th in angle_grid_deg if (th < theta1 or th > theta2)]
    for th in sidelobe_angles:
        a = steering_vector(N, th).flatten()
        pm = cp.real(cp.quad_form(a, R))
        cons.append(p0 - pm >= t)

    prob = cp.Problem(cp.Minimize(-t), cons)
    ok, used, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or R.value is None:
        raise RuntimeError(f"Radar-only shared design failed. status={status}, solver={used}")
    return R.value


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def first_null_bounds(pattern, center_idx):
    p = np.asarray(pattern).flatten()

    left = center_idx - 1
    while left > 1:
        if p[left] <= p[left - 1] and p[left] <= p[left + 1]:
            break
        left -= 1
    if left <= 1:
        left = max(center_idx - 3, 0)

    right = center_idx + 1
    while right < len(p) - 2:
        if p[right] <= p[right - 1] and p[right] <= p[right + 1]:
            break
        right += 1
    if right >= len(p) - 2:
        right = min(center_idx + 3, len(p) - 1)

    return left, right


def compute_pslr_db(pattern, angle_grid_deg, theta0=0.0):
    patt = np.maximum(np.real(pattern), 1e-15)
    patt_db = 10.0 * np.log10(patt / np.max(patt))

    center_idx = int(np.argmin(np.abs(angle_grid_deg - theta0)))
    peak_window = slice(max(center_idx - 8, 0), min(center_idx + 9, len(patt_db)))
    local_peak_idx = np.argmax(patt_db[peak_window]) + max(center_idx - 8, 0)

    left_null, right_null = first_null_bounds(patt, local_peak_idx)
    mask = np.ones(len(patt_db), dtype=bool)
    mask[left_null:right_null + 1] = False

    max_sidelobe_db = np.max(patt_db[mask])
    return -max_sidelobe_db


def average_sinr_db_from_beams(T, H, N0):
    K = T.shape[1]
    Csum = T @ herm(T)
    out = []
    for i in range(K):
        hi = H[:, i]
        desired = np.abs(hi.T @ T[:, i]) ** 2
        total = np.real(hi.T @ Csum @ np.conj(hi))
        interf = total - desired
        gam = max(float(np.real(desired / (interf + N0))), 1e-15)
        out.append(10.0 * np.log10(gam))
    return float(np.mean(out))


def average_sinr_db_from_covs(Tlist, H, N0):
    K = len(Tlist)
    Csum = sum(Tlist)
    out = []
    for i in range(K):
        hi = H[:, i]
        Qi = np.outer(np.conj(hi), hi)
        desired = np.real(np.trace(Qi @ Tlist[i]))
        interf = np.real(np.trace(Qi @ (Csum - Tlist[i])))
        gam = max(float(desired / (interf + N0)), 1e-15)
        out.append(10.0 * np.log10(gam))
    return float(np.mean(out))


def beams_from_cov_list(Tlist):
    N = Tlist[0].shape[0]
    K = len(Tlist)
    T = np.zeros((N, K), dtype=complex)
    for i, Ti in enumerate(Tlist):
        Ti = (Ti + herm(Ti)) / 2.0
        lam, U = np.linalg.eigh(Ti)
        idx = np.argmax(np.real(lam))
        val = max(float(np.real(lam[idx])), 0.0)
        T[:, i] = np.sqrt(val) * U[:, idx]
    return T


# ------------------------------------------------------------
# Constrained shared SDR, paper eq. (20)
# ------------------------------------------------------------
def solve_shared_constrained(H, R2, P0, Gamma_lin, N0, power_mode, solvers, verbose=False):
    N, K = H.shape
    T = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    C = sum(T)

    cons = []
    if power_mode == 'total':
        cons.append(cp.real(cp.trace(C)) == P0)
    elif power_mode == 'per-ant':
        cons.append(cp.diag(C) == (P0 / N) * np.ones(N))
    else:
        raise ValueError("power_mode must be 'total' or 'per-ant'")

    for k in range(K):
        cons.append(T[k] >> 0)

    for i in range(K):
        hi = H[:, i]
        Bi = np.outer(np.conj(hi), hi)
        desired = cp.real(cp.trace(Bi @ T[i]))
        interf = cp.real(cp.trace(Bi @ (C - T[i])))
        cons.append(desired >= Gamma_lin * (interf + N0))

    obj = cp.norm(C - R2, 'fro') ** 2
    prob = cp.Problem(cp.Minimize(obj), cons)
    ok, used, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or any(Ti.value is None for Ti in T):
        return None, status
    return [Ti.value for Ti in T], status


# ------------------------------------------------------------
# Exact RCG under total power: T in C^{N x K}
# ------------------------------------------------------------
def alpha_and_G_total(T, H, Gamma_lin):
    N, K = T.shape
    C = T @ herm(T)
    alpha = np.zeros(K, dtype=float)
    Gsum = np.zeros_like(T, dtype=complex)
    Gstack = []

    for i in range(K):
        hi = H[:, i]
        Bi = np.outer(np.conj(hi), hi)  # h_i^* h_i^T
        ti = T[:, i:i+1]
        desired = np.real(herm(ti) @ Bi @ ti).item()
        total = np.real(np.trace(Bi @ C))
        ai = (1.0 + Gamma_lin) * desired - Gamma_lin * total
        alpha[i] = ai

        E_i = np.zeros((1, K), dtype=complex)
        E_i[0, i] = 1.0
        Gi = Bi @ ((1.0 + Gamma_lin) * ti @ E_i - Gamma_lin * T)
        Gstack.append(Gi)

    return alpha, Gstack


def cost_grad_total(T, H, R2, Gamma_lin, rho1, rho2, penalty_type, eps_lse=0.1):
    E = T @ herm(T) - R2
    fit = np.real(np.trace(E @ E))
    grad = 4.0 * rho1 * E @ T

    alpha, Gstack = alpha_and_G_total(T, H, Gamma_lin)

    if penalty_type == 'sum-square':
        pen = np.sum(alpha ** 2)
        for i in range(len(alpha)):
            grad += 4.0 * rho2 * alpha[i] * Gstack[i]
    elif penalty_type == 'max':
        z = np.exp(-alpha / eps_lse)
        z /= np.sum(z)
        pen = eps_lse * np.log(np.sum(np.exp(-alpha / eps_lse)))
        combo = np.zeros_like(T, dtype=complex)
        for i in range(len(alpha)):
            combo += z[i] * Gstack[i]
        grad += -2.0 * rho2 * combo
    else:
        raise ValueError("penalty_type must be 'sum-square' or 'max'")

    return float(np.real(rho1 * fit + rho2 * pen)), grad


def proj_total(T, G, P0):
    coeff = np.real(np.trace(herm(T) @ G)) / max(P0, 1e-15)
    return G - coeff * T


def retract_total(Y, P0):
    nrm = np.linalg.norm(Y, 'fro')
    if nrm <= 1e-15:
        return Y.copy()
    return np.sqrt(P0) * Y / nrm


def rcg_total(T0, H, R2, P0, Gamma_lin, rho1, rho2, penalty_type,
              max_iter=300, tol=1e-7, armijo_c=1e-4, armijo_beta=0.5,
              eps_lse=0.1, verbose=False):
    T = retract_total(T0, P0)
    f, G = cost_grad_total(T, H, R2, Gamma_lin, rho1, rho2, penalty_type, eps_lse)
    grad = proj_total(T, G, P0)
    D = -grad

    def inner(A, B):
        return float(np.real(np.vdot(A, B)))

    for it in range(max_iter):
        gnorm = np.linalg.norm(grad, 'fro')
        if gnorm <= tol:
            break

        desc = inner(grad, D)
        if desc >= 0:
            D = -grad
            desc = -inner(grad, grad)

        step = 1.0
        accepted = False
        for _ in range(30):
            Tnew = retract_total(T + step * D, P0)
            fnew, Gnew = cost_grad_total(Tnew, H, R2, Gamma_lin, rho1, rho2, penalty_type, eps_lse)
            if fnew <= f + armijo_c * step * desc:
                accepted = True
                break
            step *= armijo_beta

        if not accepted:
            break

        grad_new = proj_total(Tnew, Gnew, P0)
        transported_grad = proj_total(Tnew, grad, P0)
        transported_D = proj_total(Tnew, D, P0)

        denom = max(inner(grad, grad), 1e-15)
        mu = max(0.0, inner(grad_new, grad_new - transported_grad) / denom)
        D = -grad_new + mu * transported_D

        T = Tnew
        f = fnew
        grad = grad_new

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(f"[total-{penalty_type}] iter={it:3d} cost={f:.6e} grad={np.linalg.norm(grad):.3e}")

    return T


# ------------------------------------------------------------
# Exact RCG under per-antenna constraint: X = T^H in C^{K x N}
# ------------------------------------------------------------
def alpha_and_GH_per_ant(X, H, Gamma_lin):
    K, N = X.shape
    C = herm(X) @ X
    alpha = np.zeros(K, dtype=float)
    GH_stack = []

    for i in range(K):
        hi = H[:, i]
        Bi = np.outer(np.conj(hi), hi)       # h_i^* h_i^T
        BiH = herm(Bi)                       # used in G_i^H form
        xi = X[i:i+1, :]                     # 1 x N

        desired = np.real(xi @ Bi @ herm(xi)).item()
        total = np.real(np.trace(Bi @ C))
        ai = (1.0 + Gamma_lin) * desired - Gamma_lin * total
        alpha[i] = ai

        Ei = np.zeros_like(X, dtype=complex)
        Ei[i, :] = X[i, :]
        GiH = ((1.0 + Gamma_lin) * Ei - Gamma_lin * X) @ BiH
        GH_stack.append(GiH)

    return alpha, GH_stack


def cost_grad_per_ant(X, H, R2, Gamma_lin, rho1, rho2, penalty_type, eps_lse=0.1):
    E = herm(X) @ X - R2
    fit = np.real(np.trace(E @ E))
    grad = 4.0 * rho1 * X @ E

    alpha, GH_stack = alpha_and_GH_per_ant(X, H, Gamma_lin)

    if penalty_type == 'sum-square':
        pen = np.sum(alpha ** 2)
        for i in range(len(alpha)):
            grad += 4.0 * rho2 * alpha[i] * GH_stack[i]
    elif penalty_type == 'max':
        z = np.exp(-alpha / eps_lse)
        z /= np.sum(z)
        pen = eps_lse * np.log(np.sum(np.exp(-alpha / eps_lse)))
        combo = np.zeros_like(X, dtype=complex)
        for i in range(len(alpha)):
            combo += z[i] * GH_stack[i]
        grad += -2.0 * rho2 * combo
    else:
        raise ValueError("penalty_type must be 'sum-square' or 'max'")

    return float(np.real(rho1 * fit + rho2 * pen)), grad


def proj_per_ant(X, G, P0):
    c = P0 / X.shape[1]
    D = np.real(np.diag(herm(X) @ G)) / max(c, 1e-15)
    return G - X @ np.diag(D)


def retract_per_ant(Y, P0):
    K, N = Y.shape
    target = np.sqrt(P0 / N)
    X = Y.copy()
    for n in range(N):
        cnrm = np.linalg.norm(X[:, n])
        if cnrm <= 1e-15:
            X[0, n] = target
            cnrm = np.linalg.norm(X[:, n])
        X[:, n] = target * X[:, n] / cnrm
    return X


def rcg_per_ant(X0, H, R2, P0, Gamma_lin, rho1, rho2, penalty_type,
                max_iter=300, tol=1e-7, armijo_c=1e-4, armijo_beta=0.5,
                eps_lse=0.1, verbose=False):
    X = retract_per_ant(X0, P0)
    f, G = cost_grad_per_ant(X, H, R2, Gamma_lin, rho1, rho2, penalty_type, eps_lse)
    grad = proj_per_ant(X, G, P0)
    D = -grad

    def inner(A, B):
        return float(np.real(np.vdot(A, B)))

    for it in range(max_iter):
        gnorm = np.linalg.norm(grad, 'fro')
        if gnorm <= tol:
            break

        desc = inner(grad, D)
        if desc >= 0:
            D = -grad
            desc = -inner(grad, grad)

        step = 1.0
        accepted = False
        for _ in range(30):
            Xnew = retract_per_ant(X + step * D, P0)
            fnew, Gnew = cost_grad_per_ant(Xnew, H, R2, Gamma_lin, rho1, rho2, penalty_type, eps_lse)
            if fnew <= f + armijo_c * step * desc:
                accepted = True
                break
            step *= armijo_beta

        if not accepted:
            break

        grad_new = proj_per_ant(Xnew, Gnew, P0)
        transported_grad = proj_per_ant(Xnew, grad, P0)
        transported_D = proj_per_ant(Xnew, D, P0)

        denom = max(inner(grad, grad), 1e-15)
        tau = max(0.0, inner(grad_new, grad_new - transported_grad) / denom)
        D = -grad_new + tau * transported_D

        X = Xnew
        f = fnew
        grad = grad_new

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(f"[perant-{penalty_type}] iter={it:3d} cost={f:.6e} grad={np.linalg.norm(grad):.3e}")

    return X


# ------------------------------------------------------------
# Figure 7 pipeline
# ------------------------------------------------------------
def trace_curves(seed=7,
                 gamma_grid_db=(5.8, 6.2, 6.6, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0),
                 solve_grid_step=1.0,
                 plot_grid_step=0.25,
                 verbose=False):
    N = 20
    K = 10
    P0 = dBm2W(20.0)
    N0 = dBm2W(0.0)

    rng = np.random.default_rng(seed)
    H = cn((N, K), rng)

    solvers = get_solvers()
    solve_grid = np.arange(-90.0, 90.0 + solve_grid_step, solve_grid_step)
    plot_grid = np.arange(-90.0, 90.0 + plot_grid_step, plot_grid_step)

    print("Building shared radar-only 3 dB reference ...")
    R2 = solve_radar_only_shared_3db(
        N, P0, solve_grid, theta0=0.0, beamwidth_3db=10.0,
        solvers=solvers, verbose=verbose
    )

    weights = {
        ('sum-square', 'total'):   (10.0, 1.0),
        ('sum-square', 'per-ant'): (3.0, 1.0),
        ('max',        'total'):   (10.0, 1.0),
        ('max',        'per-ant'): (1.0, 2.0),
    }

    curves = {
        ('constrained', 'total'): [],
        ('constrained', 'per-ant'): [],
        ('sum-square', 'total'): [],
        ('sum-square', 'per-ant'): [],
        ('max', 'total'): [],
        ('max', 'per-ant'): [],
    }

    for gdb in gamma_grid_db:
        Gamma_lin = db2lin(gdb)
        print(f"Gamma target = {gdb:.2f} dB")

        # constrained total
        covs_t, st = solve_shared_constrained(H, R2, P0, Gamma_lin, N0, 'total', solvers, verbose=False)
        if covs_t is not None:
            Tcov = sum(covs_t)
            x = average_sinr_db_from_covs(covs_t, H, N0)
            y = compute_pslr_db(beampattern(Tcov, plot_grid), plot_grid, 0.0)
            curves[('constrained', 'total')].append((x, y))

        # constrained per-ant
        covs_p, st = solve_shared_constrained(H, R2, P0, Gamma_lin, N0, 'per-ant', solvers, verbose=False)
        if covs_p is not None:
            Tcov = sum(covs_p)
            x = average_sinr_db_from_covs(covs_p, H, N0)
            y = compute_pslr_db(beampattern(Tcov, plot_grid), plot_grid, 0.0)
            curves[('constrained', 'per-ant')].append((x, y))

        if covs_t is None or covs_p is None:
            continue

        # use constrained SDR solutions as the initial points for weighted RCG
        T0_total = beams_from_cov_list(covs_t)
        T0_per = beams_from_cov_list(covs_p)
        X0_per = herm(T0_per)

        # sum-square total
        rho1, rho2 = weights[('sum-square', 'total')]
        T_ss_t = rcg_total(
            T0_total, H, R2, P0, Gamma_lin, rho1, rho2, 'sum-square',
            max_iter=300, tol=1e-7, eps_lse=0.1, verbose=False
        )
        x = average_sinr_db_from_beams(T_ss_t, H, N0)
        y = compute_pslr_db(beampattern(T_ss_t @ herm(T_ss_t), plot_grid), plot_grid, 0.0)
        curves[('sum-square', 'total')].append((x, y))

        # max total (warm-start from sum-square total)
        rho1, rho2 = weights[('max', 'total')]
        T_m_t = rcg_total(
            T_ss_t, H, R2, P0, Gamma_lin, rho1, rho2, 'max',
            max_iter=300, tol=1e-7, eps_lse=0.1, verbose=False
        )
        x = average_sinr_db_from_beams(T_m_t, H, N0)
        y = compute_pslr_db(beampattern(T_m_t @ herm(T_m_t), plot_grid), plot_grid, 0.0)
        curves[('max', 'total')].append((x, y))

        # sum-square per-ant
        rho1, rho2 = weights[('sum-square', 'per-ant')]
        X_ss_p = rcg_per_ant(
            X0_per, H, R2, P0, Gamma_lin, rho1, rho2, 'sum-square',
            max_iter=300, tol=1e-7, eps_lse=0.1, verbose=False
        )
        T_ss_p = herm(X_ss_p)
        x = average_sinr_db_from_beams(T_ss_p, H, N0)
        y = compute_pslr_db(beampattern(T_ss_p @ herm(T_ss_p), plot_grid), plot_grid, 0.0)
        curves[('sum-square', 'per-ant')].append((x, y))

        # max per-ant (warm-start from sum-square per-ant)
        rho1, rho2 = weights[('max', 'per-ant')]
        X_m_p = rcg_per_ant(
            X_ss_p, H, R2, P0, Gamma_lin, rho1, rho2, 'max',
            max_iter=300, tol=1e-7, eps_lse=0.1, verbose=False
        )
        T_m_p = herm(X_m_p)
        x = average_sinr_db_from_beams(T_m_p, H, N0)
        y = compute_pslr_db(beampattern(T_m_p @ herm(T_m_p), plot_grid), plot_grid, 0.0)
        curves[('max', 'per-ant')].append((x, y))

    for key in curves:
        arr = np.array(curves[key], dtype=float)
        if arr.size > 0:
            curves[key] = arr[np.argsort(arr[:, 0])]
        else:
            curves[key] = np.empty((0, 2))
    return curves


def plot_fig7(curves):
    plt.figure(figsize=(7.0, 5.9))

    cm_t = curves[('max', 'total')]
    cs_t = curves[('sum-square', 'total')]
    cc_t = curves[('constrained', 'total')]
    cm_p = curves[('max', 'per-ant')]
    cs_p = curves[('sum-square', 'per-ant')]
    cc_p = curves[('constrained', 'per-ant')]

    if len(cm_t): plt.plot(cm_t[:, 0], cm_t[:, 1], 'r-x', lw=1.8, ms=7, mew=1.5, label='Max, RCG')
    if len(cs_t): plt.plot(cs_t[:, 0], cs_t[:, 1], 'b-o', lw=1.8, ms=7, mew=1.5, fillstyle='none', label='Sum-Squ, RCG')
    if len(cc_t): plt.plot(cc_t[:, 0], cc_t[:, 1], 'k-^', lw=1.8, ms=8, mew=1.5, fillstyle='none', label='Constrained, SDR')

    if len(cm_p): plt.plot(cm_p[:, 0], cm_p[:, 1], 'r--x', lw=1.8, ms=7, mew=1.5, dashes=(4, 3))
    if len(cs_p): plt.plot(cs_p[:, 0], cs_p[:, 1], 'b--o', lw=1.8, ms=7, mew=1.5, fillstyle='none', dashes=(4, 3))
    if len(cc_p): plt.plot(cc_p[:, 0], cc_p[:, 1], 'k--^', lw=1.8, ms=8, mew=1.5, fillstyle='none', dashes=(4, 3))

    plt.title('K = 10', fontsize=18, fontweight='bold')
    plt.xlabel('Average SINR (dB)', fontsize=16)
    plt.ylabel('Average PSLR (dB)', fontsize=16)
    plt.xlim(5, 12)
    plt.ylim(9, 14)
    plt.xticks(np.arange(5, 12.1, 1), fontsize=12)
    plt.yticks(np.arange(9, 14.1, 1), fontsize=12)
    plt.grid(True, alpha=0.35)

    legend = plt.legend(loc='upper right', bbox_to_anchor=(0.86, 0.95),
                        frameon=True, fancybox=False, edgecolor='black',
                        fontsize=11, handlelength=1.3, handletextpad=0.4,
                        borderpad=0.35, labelspacing=0.25)
    plt.gca().add_artist(legend)

    plt.text(0.61, 0.62, 'Solid Lines: Total\nDashed Lines: Per-Ant',
             transform=plt.gca().transAxes, fontsize=12, va='top')

    plt.tight_layout()
    plt.savefig('Figure07_reproduced_corrected.png', dpi=300, bbox_inches='tight')
    # plt.savefig('Figure07_reproduced_corrected.pdf', bbox_inches='tight')
    plt.show()


def print_curve_points(curves):
    for key in [('max', 'total'), ('sum-square', 'total'), ('constrained', 'total'),
                ('max', 'per-ant'), ('sum-square', 'per-ant'), ('constrained', 'per-ant')]:
        print(f"\n{key}:")
        print(np.round(curves[key], 3))


def reproduce_fig7_corrected(seed=7, verbose=False):
    curves = trace_curves(seed=seed, verbose=verbose)
    plot_fig7(curves)
    print_curve_points(curves)
    return curves


if __name__ == '__main__':
    reproduce_fig7_corrected(seed=7, verbose=False)
