
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ============================================================
# Figure 7 reproduction (approximate but equation-based)
# Paper: MU-MIMO Communications With MIMO Radar: From Co-Existence to Joint Transmission
#
# What this script does:
# - Uses the shared deployment only, as stated for Fig. 7
# - Builds the 3 dB radar-only reference beampattern of the same shape as Fig. 4(b)
# - Compares:
#       * Constrained, SDR
#       * Sum-Squ weighted optimization
#       * Max weighted optimization
# - Shows both:
#       * Solid lines  : total power constraint
#       * Dashed lines : per-antenna power constraint
#
# Important honesty note:
# The paper states that Fig. 7 uses the RCG algorithm for the weighted problems
# and weighting vectors from Table II, but the exact Table II values are not
# available in the extracted text here. Therefore this script reproduces Fig. 7
# in a paper-faithful *optimization-form* way by sweeping penalty weights in the
# weighted objectives, and solving the lifted convex SDR forms directly.
# ============================================================


def db2lin(x_db):
    return 10 ** (x_db / 10.0)


def dBm2W(x_dBm):
    return 10 ** ((x_dBm - 30.0) / 10.0)


def cn(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def steering_vector(N, theta_deg, d=0.5):
    theta = np.deg2rad(theta_deg)
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * d * np.sin(theta) * n)[:, None]


def beampattern(C, angle_grid_deg):
    vals = []
    N = C.shape[0]
    for th in angle_grid_deg:
        a = steering_vector(N, th)
        vals.append(np.real((a.conj().T @ C @ a).item()))
    return np.maximum(np.array(vals), 1e-15)


def get_solvers():
    installed = set(cp.installed_solvers())
    out = []
    for s in [cp.MOSEK, cp.CLARABEL, cp.SCS]:
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


def solve_radar_only_shared_3db(N, P0, angle_grid_deg, theta0=0.0, beamwidth_3db=10.0, solvers=None, verbose=False):
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

    cons = [R >> 0, cp.diag(R) == (P0 / N) * np.ones(N), p1 == p0 / 2.0, p2 == p0 / 2.0]

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
    peak_window = slice(max(center_idx - 6, 0), min(center_idx + 7, len(patt_db)))
    local_peak_idx = np.argmax(patt_db[peak_window]) + max(center_idx - 6, 0)

    left_null, right_null = first_null_bounds(patt, local_peak_idx)
    mask = np.ones(len(patt_db), dtype=bool)
    mask[left_null:right_null + 1] = False

    max_sidelobe_db = np.max(patt_db[mask])
    return -max_sidelobe_db


def compute_user_sinr_lin(C_list, H, N0):
    K = len(C_list)
    Csum = sum(C_list)
    out = []
    for i in range(K):
        hi = H[:, i]
        Hi = np.outer(np.conj(hi), hi)
        desired = np.real(np.trace(Hi @ C_list[i]))
        interf = np.real(np.trace(Hi @ (Csum - C_list[i])))
        out.append(max(desired / (interf + N0), 1e-15))
    return np.array(out)


def average_sinr_db(C_list, H, N0):
    gam = compute_user_sinr_lin(C_list, H, N0)
    return np.mean(10.0 * np.log10(gam))


def solve_shared_constrained(H, R2, P0, Gamma_lin, N0, power_mode, solvers, verbose=False):
    N, K = H.shape
    T = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    C = sum(T)

    cons = []
    if power_mode == "total":
        cons.append(cp.real(cp.trace(C)) == P0)
    elif power_mode == "per-ant":
        cons.append(cp.diag(C) == (P0 / N) * np.ones(N))
    else:
        raise ValueError("power_mode must be 'total' or 'per-ant'")

    for k in range(K):
        cons.append(T[k] >> 0)

    for i in range(K):
        hi = H[:, i]
        Hi = np.outer(np.conj(hi), hi)
        desired = cp.real(cp.trace(Hi @ T[i]))
        interf = cp.real(cp.trace(Hi @ (C - T[i])))
        cons.append(desired >= Gamma_lin * (interf + N0))

    obj = cp.norm(C - R2, 'fro') ** 2
    prob = cp.Problem(cp.Minimize(obj), cons)
    ok, used, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or any(Ti.value is None for Ti in T):
        return None, status
    return [Ti.value for Ti in T], status


def solve_shared_weighted(H, R2, P0, Gamma_ref_lin, N0, rho1, rho2, penalty_type, power_mode, solvers, verbose=False):
    N, K = H.shape
    T = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    C = sum(T)

    cons = []
    if power_mode == "total":
        cons.append(cp.real(cp.trace(C)) == P0)
    elif power_mode == "per-ant":
        cons.append(cp.diag(C) == (P0 / N) * np.ones(N))
    else:
        raise ValueError("power_mode must be 'total' or 'per-ant'")

    for k in range(K):
        cons.append(T[k] >> 0)

    alpha_expr = []
    for i in range(K):
        hi = H[:, i]
        Hi = np.outer(np.conj(hi), hi)
        desired = cp.real(cp.trace(Hi @ T[i]))
        total = cp.real(cp.trace(Hi @ C))
        alpha_i = (1.0 + Gamma_ref_lin) * desired - Gamma_ref_lin * total
        alpha_expr.append(alpha_i)

    alpha_vec = cp.hstack(alpha_expr)
    fit_term = cp.norm(C - R2, 'fro') ** 2

    if penalty_type == "sum-square":
        pen = cp.sum_squares(alpha_vec - (N0 * Gamma_ref_lin) * np.ones(K))
    elif penalty_type == "max":
        pen = cp.max(-alpha_vec)
    else:
        raise ValueError("penalty_type must be 'sum-square' or 'max'")

    obj = rho1 * fit_term + rho2 * pen
    prob = cp.Problem(cp.Minimize(obj), cons)
    ok, used, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or any(Ti.value is None for Ti in T):
        return None, status
    return [Ti.value for Ti in T], status


def average_curve_constrained(K=10, n_mc=8, gamma_targets_db=None, seed=7, angle_step_deg=1.0, verbose=False):
    if gamma_targets_db is None:
        gamma_targets_db = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

    N = 20
    P0 = dBm2W(20.0)
    N0 = dBm2W(0.0)
    angle_grid = np.arange(-90.0, 90.0 + angle_step_deg, angle_step_deg)

    solvers = get_solvers()
    rng = np.random.default_rng(seed)

    out_total = []
    out_per = []

    for gdb in gamma_targets_db:
        glin = db2lin(gdb)
        sinr_total, pslr_total = [], []
        sinr_per, pslr_per = [], []

        print(f"[Constrained] target Gamma = {gdb:.1f} dB")
        for mc in range(n_mc):
            H = cn((N, K), rng)
            R2 = solve_radar_only_shared_3db(N, P0, angle_grid, 0.0, 10.0, solvers, verbose)

            sol_total, _ = solve_shared_constrained(H, R2, P0, glin, N0, "total", solvers, verbose)
            if sol_total is not None:
                sinr_total.append(average_sinr_db(sol_total, H, N0))
                pslr_total.append(compute_pslr_db(beampattern(sum(sol_total), angle_grid), angle_grid, 0.0))

            sol_per, _ = solve_shared_constrained(H, R2, P0, glin, N0, "per-ant", solvers, verbose)
            if sol_per is not None:
                sinr_per.append(average_sinr_db(sol_per, H, N0))
                pslr_per.append(compute_pslr_db(beampattern(sum(sol_per), angle_grid), angle_grid, 0.0))

        out_total.append((np.mean(sinr_total), np.mean(pslr_total)))
        out_per.append((np.mean(sinr_per), np.mean(pslr_per)))

    return np.array(out_total), np.array(out_per)


def average_curve_weighted(K=10, n_mc=8, gamma_ref_db=10.0, rho2_grid=None, penalty_type="sum-square",
                           seed=77, angle_step_deg=1.0, verbose=False):
    if rho2_grid is None:
        rho2_grid = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    N = 20
    P0 = dBm2W(20.0)
    N0 = dBm2W(0.0)
    angle_grid = np.arange(-90.0, 90.0 + angle_step_deg, angle_step_deg)
    Gamma_ref_lin = db2lin(gamma_ref_db)
    rho1 = 1.0

    solvers = get_solvers()
    rng = np.random.default_rng(seed)

    out_total = []
    out_per = []

    for rho2 in rho2_grid:
        sinr_total, pslr_total = [], []
        sinr_per, pslr_per = [], []

        print(f"[{penalty_type}] rho2 = {rho2}")
        for mc in range(n_mc):
            H = cn((N, K), rng)
            R2 = solve_radar_only_shared_3db(N, P0, angle_grid, 0.0, 10.0, solvers, verbose)

            sol_total, _ = solve_shared_weighted(
                H, R2, P0, Gamma_ref_lin, N0, rho1, rho2, penalty_type, "total", solvers, verbose
            )
            if sol_total is not None:
                sinr_total.append(average_sinr_db(sol_total, H, N0))
                pslr_total.append(compute_pslr_db(beampattern(sum(sol_total), angle_grid), angle_grid, 0.0))

            sol_per, _ = solve_shared_weighted(
                H, R2, P0, Gamma_ref_lin, N0, rho1, rho2, penalty_type, "per-ant", solvers, verbose
            )
            if sol_per is not None:
                sinr_per.append(average_sinr_db(sol_per, H, N0))
                pslr_per.append(compute_pslr_db(beampattern(sum(sol_per), angle_grid), angle_grid, 0.0))

        out_total.append((np.mean(sinr_total), np.mean(pslr_total)))
        out_per.append((np.mean(sinr_per), np.mean(pslr_per)))

    arr_total = np.array(out_total)
    arr_per = np.array(out_per)

    idx_t = np.argsort(arr_total[:, 0])
    idx_p = np.argsort(arr_per[:, 0])
    return arr_total[idx_t], arr_per[idx_p]


def plot_fig7(curve_max_total, curve_max_per, curve_ss_total, curve_ss_per, curve_c_total, curve_c_per):
    plt.figure(figsize=(7.0, 5.9))

    plt.plot(curve_max_total[:, 0], curve_max_total[:, 1], 'r-x', lw=1.8, ms=7, mew=1.5, label='Max, RCG')
    plt.plot(curve_ss_total[:, 0], curve_ss_total[:, 1], 'b-o', lw=1.8, ms=7, mew=1.5, fillstyle='none', label='Sum-Squ, RCG')
    plt.plot(curve_c_total[:, 0], curve_c_total[:, 1], 'k-^', lw=1.8, ms=8, mew=1.5, fillstyle='none', label='Constrained, SDR')

    plt.plot(curve_max_per[:, 0], curve_max_per[:, 1], 'r--x', lw=1.8, ms=7, mew=1.5, dashes=(4, 3))
    plt.plot(curve_ss_per[:, 0], curve_ss_per[:, 1], 'b--o', lw=1.8, ms=7, mew=1.5, fillstyle='none', dashes=(4, 3))
    plt.plot(curve_c_per[:, 0], curve_c_per[:, 1], 'k--^', lw=1.8, ms=8, mew=1.5, fillstyle='none', dashes=(4, 3))

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

    plt.text(0.63, 0.63, 'Solid Lines: Total\nDashed Lines: Per-Ant',
             transform=plt.gca().transAxes, fontsize=12, va='top')

    plt.tight_layout()
    # plt.savefig('/mnt/data/Figure07_reproduced.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def reproduce_fig7_fast():
    curve_c_total, curve_c_per = average_curve_constrained(
        K=10, n_mc=4,
        gamma_targets_db=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        seed=7, angle_step_deg=2.0, verbose=False
    )

    curve_ss_total, curve_ss_per = average_curve_weighted(
        K=10, n_mc=4, gamma_ref_db=10.0,
        rho2_grid=[0.03, 0.06, 0.12, 0.25, 0.5, 1.0],
        penalty_type='sum-square',
        seed=77, angle_step_deg=2.0, verbose=False
    )

    curve_max_total, curve_max_per = average_curve_weighted(
        K=10, n_mc=4, gamma_ref_db=10.0,
        rho2_grid=[0.01, 0.03, 0.07, 0.15, 0.3, 0.7],
        penalty_type='max',
        seed=177, angle_step_deg=2.0, verbose=False
    )

    plot_fig7(curve_max_total, curve_max_per, curve_ss_total, curve_ss_per, curve_c_total, curve_c_per)

    print("Constrained total:\n", np.round(curve_c_total, 3))
    print("Constrained per-ant:\n", np.round(curve_c_per, 3))
    print("Sum-square total:\n", np.round(curve_ss_total, 3))
    print("Sum-square per-ant:\n", np.round(curve_ss_per, 3))
    print("Max total:\n", np.round(curve_max_total, 3))
    print("Max per-ant:\n", np.round(curve_max_per, 3))


def reproduce_fig7_paperlike():
    curve_c_total, curve_c_per = average_curve_constrained(
        K=10, n_mc=12,
        gamma_targets_db=[5.8, 6.5, 7.1, 8.0, 9.0, 10.0, 11.0],
        seed=7, angle_step_deg=1.0, verbose=False
    )

    curve_ss_total, curve_ss_per = average_curve_weighted(
        K=10, n_mc=12, gamma_ref_db=10.0,
        rho2_grid=[0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28],
        penalty_type='sum-square',
        seed=77, angle_step_deg=1.0, verbose=False
    )

    curve_max_total, curve_max_per = average_curve_weighted(
        K=10, n_mc=12, gamma_ref_db=10.0,
        rho2_grid=[0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80],
        penalty_type='max',
        seed=177, angle_step_deg=1.0, verbose=False
    )

    plot_fig7(curve_max_total, curve_max_per, curve_ss_total, curve_ss_per, curve_c_total, curve_c_per)

    print("Constrained total:\n", np.round(curve_c_total, 3))
    print("Constrained per-ant:\n", np.round(curve_c_per, 3))
    print("Sum-square total:\n", np.round(curve_ss_total, 3))
    print("Sum-square per-ant:\n", np.round(curve_ss_per, 3))
    print("Max total:\n", np.round(curve_max_total, 3))
    print("Max per-ant:\n", np.round(curve_max_per, 3))


if __name__ == "__main__":
    reproduce_fig7_fast()
