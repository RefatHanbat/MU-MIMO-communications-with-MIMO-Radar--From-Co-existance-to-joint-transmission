
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# ============================================================
# Figure07_reproduce_v2.py
# Paper-faithful rebuild for Fig. 7 of:
# Fan Liu et al., "MU-MIMO Communications With MIMO Radar:
# From Co-Existence to Joint Transmission", IEEE TWC, 2018.
#
# Key fixes vs earlier versions:
#   1) Shared deployment only
#   2) Same 3 dB radar-only reference shape as Fig. 4(b)
#   3) Weighted methods use the paper's Table II weights:
#        - Sum-Square, total    [rho1, rho2] = [10, 1]
#        - Sum-Square, per-ant  [rho1, rho2] = [3, 1]
#        - Max, total           [rho1, rho2] = [10, 1]
#        - Max, per-ant         [rho1, rho2] = [1, 2]
#   4) Trade-off is traced by varying the target Gamma (dB),
#      not by arbitrarily sweeping penalty weights.
#   5) Weighted solvers are direct manifold-style RCG solvers
#      on the beamforming matrix, not lifted SDR surrogates.
#
# Honest note:
#   Exact pixel-by-pixel duplication is still not guaranteed,
#   because the paper does not publish the exact channel draws.
#   This script fixes the methodology mistakes and reproduces
#   Fig. 7 in a much more paper-faithful way.
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


# -----------------------------
# Radar-only reference (paper eq. (10))
# -----------------------------
def solve_radar_only_shared_3db(N, P0, angle_grid_deg, theta0=0.0, beamwidth_3db=10.0,
                                solvers=None, verbose=False):
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
        raise RuntimeError(f"Radar-only design failed. status={status}, solver={used}")
    return R.value


# -----------------------------
# Utilities
# -----------------------------
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


def average_sinr_db_cov(C_list, H, N0):
    Csum = sum(C_list)
    K = len(C_list)
    gam_db = []
    for i in range(K):
        hi = H[:, i]
        Qi = np.outer(np.conj(hi), hi)
        desired = np.real(np.trace(Qi @ C_list[i]))
        interf = np.real(np.trace(Qi @ (Csum - C_list[i])))
        gam = max(desired / (interf + N0), 1e-15)
        gam_db.append(10.0 * np.log10(gam))
    return float(np.mean(gam_db))


def average_sinr_db_beam(X, H, N0):
    Csum = X @ herm(X)
    K = X.shape[1]
    gam_db = []
    for i in range(K):
        hi = H[:, i]
        desired = np.abs(hi.T @ X[:, i]) ** 2
        total = np.real(hi.T @ Csum @ np.conj(hi))
        interf = total - desired
        gam = max(float(np.real(desired / (interf + N0))), 1e-15)
        gam_db.append(10.0 * np.log10(gam))
    return float(np.mean(gam_db))


def factor_from_cov(R, K):
    lam, U = np.linalg.eigh((R + herm(R)) / 2.0)
    idx = np.argsort(lam)[::-1]
    lam = np.maximum(lam[idx], 0.0)
    U = U[:, idx]
    r = min(K, len(lam))
    X = U[:, :r] @ np.diag(np.sqrt(lam[:r]))
    if r < K:
        X = np.hstack([X, np.zeros((R.shape[0], K - r), dtype=complex)])
    return X


def project_total(X, G, P0):
    coeff = np.real(np.vdot(X, G)) / max(P0, 1e-15)
    return G - coeff * X


def retract_total(Y, P0):
    nrm = np.linalg.norm(Y, 'fro')
    if nrm <= 1e-15:
        return Y
    return np.sqrt(P0) * Y / nrm


def transport_total(Xnew, Z, P0):
    return project_total(Xnew, Z, P0)


def project_per_ant(X, G):
    P = np.zeros_like(G)
    for n in range(X.shape[0]):
        xn = X[n, :]
        gn = G[n, :]
        denom = np.real(np.vdot(xn, xn))
        coeff = 0.0 if denom <= 1e-15 else np.real(np.vdot(xn, gn)) / denom
        P[n, :] = gn - coeff * xn
    return P


def retract_per_ant(Y, target_row_norm):
    Z = Y.copy()
    for n in range(Z.shape[0]):
        rn = np.linalg.norm(Z[n, :])
        if rn <= 1e-15:
            Z[n, 0] = target_row_norm
            rn = np.linalg.norm(Z[n, :])
        Z[n, :] = target_row_norm * Z[n, :] / rn
    return Z


def transport_per_ant(Xnew, Z):
    return project_per_ant(Xnew, Z)


def alpha_vector(X, H, Gamma_lin):
    N, K = X.shape
    Csum = X @ herm(X)
    alpha = np.zeros(K, dtype=float)
    Q_list = []
    for i in range(K):
        hi = H[:, i]
        Qi = np.outer(np.conj(hi), hi)
        Q_list.append(Qi)
        desired = np.real(herm(X[:, i:i+1]) @ Qi @ X[:, i:i+1]).item()
        total = np.real(np.trace(Qi @ Csum))
        alpha[i] = (1.0 + Gamma_lin) * desired - Gamma_lin * total
    return alpha, Q_list


def cost_grad_weighted(X, H, R2, Gamma_lin, N0, rho1, rho2, penalty_type, tau=0.12):
    N, K = X.shape
    C = X @ herm(X)
    E = C - R2
    fit = np.real(np.trace(E @ E))
    G_fit = 4.0 * E @ X

    alpha, Q_list = alpha_vector(X, H, Gamma_lin)

    if penalty_type == 'sum-square':
        c = alpha - Gamma_lin * N0
        pen = np.sum(c ** 2)
        M = np.zeros((N, N), dtype=complex)
        for i in range(K):
            M += c[i] * Q_list[i]

        G_pen = np.zeros_like(X)
        for j in range(K):
            G_pen[:, j] = 4.0 * (
                -Gamma_lin * (M @ X[:, j])
                + (1.0 + Gamma_lin) * c[j] * (Q_list[j] @ X[:, j])
            )

    elif penalty_type == 'max':
        z = -alpha / tau
        z0 = np.max(z)
        ez = np.exp(z - z0)
        soft = ez / np.sum(ez)
        pen = tau * (z0 + np.log(np.sum(ez)))

        M = np.zeros((N, N), dtype=complex)
        for i in range(K):
            M += soft[i] * Q_list[i]

        G_pen = np.zeros_like(X)
        for j in range(K):
            G_pen[:, j] = 2.0 * (
                Gamma_lin * (M @ X[:, j])
                - (1.0 + Gamma_lin) * soft[j] * (Q_list[j] @ X[:, j])
            )
    else:
        raise ValueError("penalty_type must be 'sum-square' or 'max'")

    cost = rho1 * fit + rho2 * pen
    grad = rho1 * G_fit + rho2 * G_pen
    return float(np.real(cost)), grad


def rcg_weighted(H, R2, P0, Gamma_lin, N0, rho1, rho2, penalty_type, power_mode,
                 X0=None, max_iter=300, tol=1e-7, tau=0.12, verbose=False):
    N, K = H.shape

    if X0 is None:
        X = factor_from_cov(R2, K)
    else:
        X = X0.copy()

    if power_mode == 'total':
        X = retract_total(X, P0)
        project = lambda Xcur, Gcur: project_total(Xcur, Gcur, P0)
        retract = lambda Y: retract_total(Y, P0)
        transport = lambda Xnew, Z: transport_total(Xnew, Z, P0)
    elif power_mode == 'per-ant':
        row_norm = np.sqrt(P0 / N)
        X = retract_per_ant(X, row_norm)
        project = lambda Xcur, Gcur: project_per_ant(Xcur, Gcur)
        retract = lambda Y: retract_per_ant(Y, row_norm)
        transport = lambda Xnew, Z: transport_per_ant(Xnew, Z)
    else:
        raise ValueError("power_mode must be 'total' or 'per-ant'")

    f, G = cost_grad_weighted(X, H, R2, Gamma_lin, N0, rho1, rho2, penalty_type, tau=tau)
    grad = project(X, G)
    D = -grad

    def inner(A, B):
        return float(np.real(np.vdot(A, B)))

    for it in range(max_iter):
        gnorm = np.linalg.norm(grad)
        if gnorm <= tol:
            break

        step = 1.0
        armijo = 1e-4
        beta = 0.5
        desc = inner(grad, D)
        if desc >= 0:
            D = -grad
            desc = -inner(grad, grad)

        accepted = False
        for _ in range(25):
            Xcand = retract(X + step * D)
            fcand, Gcand = cost_grad_weighted(Xcand, H, R2, Gamma_lin, N0, rho1, rho2,
                                             penalty_type, tau=tau)
            if fcand <= f + armijo * step * desc:
                accepted = True
                break
            step *= beta

        if not accepted:
            break

        grad_new = project(Xcand, Gcand)
        yk = grad_new - transport(Xcand, grad)
        denom = max(inner(grad, grad), 1e-15)
        beta_pr = max(0.0, inner(grad_new, yk) / denom)

        D = -grad_new + beta_pr * transport(Xcand, D)

        X = Xcand
        f = fcand
        grad = grad_new

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(f"  iter={it:3d} cost={f:.6e} grad={np.linalg.norm(grad):.3e}")

    return X


# -----------------------------
# Constrained SDR shared deployment
# -----------------------------
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
        Qi = np.outer(np.conj(hi), hi)
        desired = cp.real(cp.trace(Qi @ T[i]))
        interf = cp.real(cp.trace(Qi @ (C - T[i])))
        cons.append(desired >= Gamma_lin * (interf + N0))

    obj = cp.norm(C - R2, 'fro') ** 2
    prob = cp.Problem(cp.Minimize(obj), cons)
    ok, used, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or any(Ti.value is None for Ti in T):
        return None, status
    return [Ti.value for Ti in T], status


# -----------------------------
# Main experiment
# -----------------------------
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
    R2 = solve_radar_only_shared_3db(N, P0, solve_grid, theta0=0.0, beamwidth_3db=10.0,
                                     solvers=solvers, verbose=verbose)

    # Table II weights
    WGT = {
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

    # Warm starts for weighted methods
    Xinit_total = factor_from_cov(R2, K)
    Xinit_per = factor_from_cov(R2, K)

    for gdb in gamma_grid_db:
        Gamma_lin = db2lin(gdb)
        print(f"Gamma target = {gdb:.2f} dB")

        # constrained total
        sol_ct, st = solve_shared_constrained(H, R2, P0, Gamma_lin, N0, 'total', solvers, verbose=False)
        if sol_ct is not None:
            x = average_sinr_db_cov(sol_ct, H, N0)
            y = compute_pslr_db(beampattern(sum(sol_ct), plot_grid), plot_grid, 0.0)
            curves[('constrained', 'total')].append((x, y))

        # constrained per-ant
        sol_cp, st = solve_shared_constrained(H, R2, P0, Gamma_lin, N0, 'per-ant', solvers, verbose=False)
        if sol_cp is not None:
            x = average_sinr_db_cov(sol_cp, H, N0)
            y = compute_pslr_db(beampattern(sum(sol_cp), plot_grid), plot_grid, 0.0)
            curves[('constrained', 'per-ant')].append((x, y))

        # sum-square total
        rho1, rho2 = WGT[('sum-square', 'total')]
        X_ss_t = rcg_weighted(H, R2, P0, Gamma_lin, N0, rho1, rho2, 'sum-square', 'total',
                              X0=Xinit_total, max_iter=250, tol=1e-7, tau=0.12, verbose=False)
        Xinit_total = X_ss_t.copy()
        x = average_sinr_db_beam(X_ss_t, H, N0)
        y = compute_pslr_db(beampattern(X_ss_t @ herm(X_ss_t), plot_grid), plot_grid, 0.0)
        curves[('sum-square', 'total')].append((x, y))

        # sum-square per-ant
        rho1, rho2 = WGT[('sum-square', 'per-ant')]
        X_ss_p = rcg_weighted(H, R2, P0, Gamma_lin, N0, rho1, rho2, 'sum-square', 'per-ant',
                              X0=Xinit_per, max_iter=250, tol=1e-7, tau=0.12, verbose=False)
        Xinit_per = X_ss_p.copy()
        x = average_sinr_db_beam(X_ss_p, H, N0)
        y = compute_pslr_db(beampattern(X_ss_p @ herm(X_ss_p), plot_grid), plot_grid, 0.0)
        curves[('sum-square', 'per-ant')].append((x, y))

        # max total: warm-start from current sum-square total for same Gamma
        rho1, rho2 = WGT[('max', 'total')]
        X_m_t = rcg_weighted(H, R2, P0, Gamma_lin, N0, rho1, rho2, 'max', 'total',
                             X0=X_ss_t, max_iter=250, tol=1e-7, tau=0.12, verbose=False)
        x = average_sinr_db_beam(X_m_t, H, N0)
        y = compute_pslr_db(beampattern(X_m_t @ herm(X_m_t), plot_grid), plot_grid, 0.0)
        curves[('max', 'total')].append((x, y))

        # max per-ant
        rho1, rho2 = WGT[('max', 'per-ant')]
        X_m_p = rcg_weighted(H, R2, P0, Gamma_lin, N0, rho1, rho2, 'max', 'per-ant',
                             X0=X_ss_p, max_iter=250, tol=1e-7, tau=0.12, verbose=False)
        x = average_sinr_db_beam(X_m_p, H, N0)
        y = compute_pslr_db(beampattern(X_m_p @ herm(X_m_p), plot_grid), plot_grid, 0.0)
        curves[('max', 'per-ant')].append((x, y))

    # sort by x-axis for cleaner curves
    for key in list(curves.keys()):
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
    plt.savefig('Figure07_reproduced_v2.png', dpi=300, bbox_inches='tight')
    # plt.savefig('Figure07_reproduced_v2.pdf', bbox_inches='tight')
    plt.show()


def print_curve_points(curves):
    for key in [('max', 'total'), ('sum-square', 'total'), ('constrained', 'total'),
                ('max', 'per-ant'), ('sum-square', 'per-ant'), ('constrained', 'per-ant')]:
        arr = curves[key]
        print(f"\n{key}:")
        print(np.round(arr, 3))


def reproduce_fig7_v2(seed=7, verbose=False):
    curves = trace_curves(seed=seed, verbose=verbose)
    plot_fig7(curves)
    print_curve_points(curves)
    return curves


if __name__ == '__main__':
    reproduce_fig7_v2(seed=7, verbose=False)
