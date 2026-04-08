
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ============================================================
# Fig. 6 reproduction for:
# Fan Liu et al., "MU-MIMO Communications With MIMO Radar:
# From Co-Existence to Joint Transmission", IEEE TWC, 2018.
#
# What Fig. 6 shows:
#   - feasibility probability of constrained problems vs penalty problems
#   - Gamma = 10 dB
#   - K in {17, 18, 19, 20}
#
# Paper-faithful interpretation:
#   1) "Separated, SDR"  -> feasibility of constrained separated problem (19)
#   2) "Shared, SDR"     -> feasibility of constrained shared problem (20)
#   3) "Shared, RCG"     -> weighted shared problem on manifold
#
# Key point from the paper:
#   "all the weighted optimizations are always feasible"
# so the RCG bar is 100% for every K.  The two SDR bars are estimated
# by Monte Carlo feasibility tests over random Rayleigh channels.
#
# Practical note:
# Exact percentages depend on Monte Carlo count, random seed, solver
# tolerance, and channel draws. The paper gives the trend and approximate
# percentages but not the exact random realizations. So this script
# reproduces the methodology and should give very similar behavior,
# not guaranteed pixel-identical bar heights.
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


def solve_radar_only_shared_3db(N, P0, angle_grid_deg, theta0, beamwidth_3db, solvers, verbose=False):
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
    ok, _, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or R.value is None:
        return None, status
    return R.value, status


def solve_radar_only_separated_3db(NR, PR, angle_grid_deg, f_list, theta0, beamwidth_3db, solvers, verbose=False):
    theta1 = theta0 - beamwidth_3db / 2.0
    theta2 = theta0 + beamwidth_3db / 2.0

    R1 = cp.Variable((NR, NR), hermitian=True)
    t = cp.Variable()

    a0 = steering_vector(NR, theta0).flatten()
    a1 = steering_vector(NR, theta1).flatten()
    a2 = steering_vector(NR, theta2).flatten()

    p0 = cp.real(cp.quad_form(a0, R1))
    p1 = cp.real(cp.quad_form(a1, R1))
    p2 = cp.real(cp.quad_form(a2, R1))

    cons = [R1 >> 0, cp.diag(R1) == (PR / NR) * np.ones(NR), p1 == p0 / 2.0, p2 == p0 / 2.0]

    for f in f_list:
        Fi = np.outer(np.conj(f), f)
        cons.append(cp.real(cp.trace(Fi @ R1)) == 0)

    sidelobe_angles = [th for th in angle_grid_deg if (th < theta1 or th > theta2)]
    for th in sidelobe_angles:
        a = steering_vector(NR, th).flatten()
        pm = cp.real(cp.quad_form(a, R1))
        cons.append(p0 - pm >= t)

    prob = cp.Problem(cp.Minimize(-t), cons)
    ok, _, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or R1.value is None:
        return None, status
    return R1.value, status


def check_shared_sdr_feasible(N, K, H, R2, P0, Gamma_lin, N0, solvers, verbose=False):
    T = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    C = sum(T)

    cons = [cp.diag(C) == (P0 / N) * np.ones(N)]
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
    ok, _, _ = solve_with_fallback(prob, solvers, verbose)
    return ok


def check_separated_sdr_feasible(NC, K, G, R1, angle_grid_deg, PC, Gamma_lin, N0, solvers, verbose=False):
    W = [cp.Variable((NC, NC), hermitian=True) for _ in range(K)]
    sigma = cp.Variable(nonneg=True)
    Cc = sum(W)

    A1 = np.hstack([steering_vector(R1.shape[0], th) for th in angle_grid_deg])
    A2 = np.hstack([steering_vector(NC, th) for th in angle_grid_deg])

    radar_diag = np.real(np.diag(A1.conj().T @ R1 @ A1))
    comm_diag = cp.hstack([cp.real(cp.quad_form(A2[:, m], Cc)) for m in range(A2.shape[1])])
    obj = cp.sum_squares(comm_diag - sigma * radar_diag)

    cons = [cp.real(cp.trace(Cc)) <= PC]
    for k in range(K):
        cons.append(W[k] >> 0)

    for i in range(K):
        gi = G[:, i]
        Gi = np.outer(np.conj(gi), gi)
        desired = cp.real(cp.trace(Gi @ W[i]))
        interf = cp.real(cp.trace(Gi @ (Cc - W[i])))
        cons.append(desired >= Gamma_lin * (interf + N0))

    prob = cp.Problem(cp.Minimize(obj), cons)
    ok, _, _ = solve_with_fallback(prob, solvers, verbose)
    return ok


def monte_carlo_feasibility(K_list, n_mc=50, seed=7, angle_step_deg=1.0, beamwidth_3db=10.0, verbose=False):
    N = 20
    P0 = dBm2W(20.0)
    N0 = dBm2W(0.0)
    Gamma_lin = db2lin(10.0)

    NR = 14
    NC = 6
    PR = P0 / 2.0
    PC = P0 / 2.0

    theta0 = 0.0
    angle_grid = np.arange(-90.0, 90.0 + angle_step_deg, angle_step_deg)

    solvers = get_solvers()
    rng = np.random.default_rng(seed)

    separated_pct = []
    shared_pct = []
    rcg_pct = []

    for K in K_list:
        sep_ok = 0
        shared_ok = 0

        print(f"\nK = {K}")
        for mc in range(n_mc):
            if (mc + 1) % 5 == 0 or mc == 0:
                print(f"  MC {mc+1}/{n_mc}")

            H = cn((N, K), rng)
            F = H[:NR, :]
            G = H[NR:, :]

            R1, _ = solve_radar_only_separated_3db(
                NR, PR, angle_grid, [F[:, i] for i in range(K)],
                theta0, beamwidth_3db, solvers, verbose
            )
            if R1 is not None:
                if check_separated_sdr_feasible(NC, K, G, R1, angle_grid, PC, Gamma_lin, N0, solvers, verbose):
                    sep_ok += 1

            R2, _ = solve_radar_only_shared_3db(
                N, P0, angle_grid, theta0, beamwidth_3db, solvers, verbose
            )
            if R2 is not None:
                if check_shared_sdr_feasible(N, K, H, R2, P0, Gamma_lin, N0, solvers, verbose):
                    shared_ok += 1

        separated_pct.append(100.0 * sep_ok / n_mc)
        shared_pct.append(100.0 * shared_ok / n_mc)
        rcg_pct.append(100.0)

        print(f"  Separated, SDR feasibility = {separated_pct[-1]:.1f}%")
        print(f"  Shared, SDR feasibility    = {shared_pct[-1]:.1f}%")
        print(f"  Shared, RCG feasibility    = {rcg_pct[-1]:.1f}%")

    return np.array(separated_pct), np.array(shared_pct), np.array(rcg_pct)


def plot_fig6(K_list, separated_pct, shared_pct, rcg_pct):
    x = np.arange(len(K_list))
    w = 0.18

    plt.figure(figsize=(6.0, 5.8))
    plt.bar(x - w, separated_pct, width=w, color='k', edgecolor='k', label='Separated, SDR')
    plt.bar(x,      shared_pct,    width=w, color='#1f77b4', edgecolor='k', linewidth=0.8, label='Shared, SDR')
    plt.bar(x + w,  rcg_pct,       width=w, color='#b30d2f', edgecolor='k', linewidth=0.8, label='Shared, RCG')

    plt.ylabel('Feasible Possibility (%)', fontsize=15)
    plt.xlabel('Users (K)', fontsize=15)
    plt.xticks(x, [str(k) for k in K_list], fontsize=12)
    plt.yticks(np.arange(0, 121, 20), fontsize=12)
    plt.ylim(0, 120)
    plt.xlim(-0.6, len(K_list) - 0.1)
    plt.grid(True, axis='y', alpha=0.35)
    plt.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=11)
    plt.tight_layout()
    plt.savefig('/mnt/data/Figure06_reproduced.png', dpi=300, bbox_inches='tight')
    plt.savefig('/mnt/data/Figure06_reproduced.pdf', bbox_inches='tight')
    plt.show()


def reproduce_fig6_fast():
    K_list = [17, 18, 19, 20]
    separated_pct, shared_pct, rcg_pct = monte_carlo_feasibility(
        K_list=K_list,
        n_mc=25,
        seed=7,
        angle_step_deg=2.0,
        beamwidth_3db=10.0,
        verbose=False,
    )
    plot_fig6(K_list, separated_pct, shared_pct, rcg_pct)
    return separated_pct, shared_pct, rcg_pct


def reproduce_fig6_paperlike():
    K_list = [17, 18, 19, 20]
    separated_pct, shared_pct, rcg_pct = monte_carlo_feasibility(
        K_list=K_list,
        n_mc=100,
        seed=7,
        angle_step_deg=1.0,
        beamwidth_3db=10.0,
        verbose=False,
    )
    plot_fig6(K_list, separated_pct, shared_pct, rcg_pct)
    return separated_pct, shared_pct, rcg_pct


if __name__ == "__main__":
    reproduce_fig6_fast()
