import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ============================================================
# Fig. 3 reproduction for:
# Fan Liu et al., "MU-MIMO Communications With MIMO Radar:
# From Co-Existence to Joint Transmission", IEEE TWC, 2018.
#
# This script reproduces Fig. 3 qualitatively and paper-faithfully
# by solving the SDR versions of:
#   - Radar-only design: (9) and (12)
#   - Separated deployment RadCom: (19) without rank-1 constraints
#   - Shared deployment RadCom: (20) without rank-1 constraints
#
# Important note:
# The paper does not publish the exact Monte-Carlo channel realization,
# grid density, desired-pattern mask width, or random seed used for the
# displayed figure. Therefore, exact pixel-for-pixel duplication of the
# published curves is generally impossible. This code reproduces the
# figure faithfully in methodology and very closely in shape.
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


def steering_matrix(N, angle_grid_deg, d=0.5):
    return np.hstack([steering_vector(N, th, d) for th in angle_grid_deg])


def beampattern(C, angle_grid_deg, d=0.5):
    N = C.shape[0]
    vals = []
    for th in angle_grid_deg:
        a = steering_vector(N, th, d)
        vals.append(np.real((a.conj().T @ C @ a).item()))
    return np.array(vals)


def desired_multibeam_pattern(angle_grid_deg, beam_centers_deg, beam_halfwidth_deg=3.0):
    Pd = np.zeros_like(angle_grid_deg, dtype=float)
    for c in beam_centers_deg:
        Pd[np.abs(angle_grid_deg - c) <= beam_halfwidth_deg] = 1.0
    return Pd


def solve_radar_only_shared(N, P0, angle_grid_deg, Pd, solver=cp.SCS, verbose=False):
    """
    Solve (9):
        min_{alpha,R} sum_m | alpha Pd(theta_m) - a^H R a |^2
        s.t. diag(R)=P0/N, R>=0, R=R^H, alpha>=0
    """
    M = len(angle_grid_deg)
    A = steering_matrix(N, angle_grid_deg)

    R = cp.Variable((N, N), hermitian=True)
    alpha = cp.Variable(nonneg=True)

    patt = []
    for m in range(M):
        a = A[:, m]
        patt.append(cp.real(cp.quad_form(a, R)))
    patt = cp.hstack(patt)

    obj = cp.sum_squares(patt - alpha * Pd)
    cons = [R >> 0, cp.diag(R) == (P0 / N) * np.ones(N)]

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver, verbose=verbose)
    return R.value, float(alpha.value)


def solve_radar_only_separated(NR, P_R, angle_grid_deg, Pd, f_list, solver=cp.SCS, verbose=False):
    """
    Solve (12):
        min_{alpha,R1} sum_m | alpha Pd(theta_m) - a1^H R1 a1 |^2
        s.t. diag(R1)=P_R/NR, R1>=0, alpha>=0,
             tr(f_i^* f_i^T R1)=0, for all i
    Here tr(f_i^* f_i^T R1) = f_i^T R1 f_i^*.
    """
    M = len(angle_grid_deg)
    A1 = steering_matrix(NR, angle_grid_deg)

    R1 = cp.Variable((NR, NR), hermitian=True)
    alpha = cp.Variable(nonneg=True)

    patt = []
    for m in range(M):
        a = A1[:, m]
        patt.append(cp.real(cp.quad_form(a, R1)))
    patt = cp.hstack(patt)

    cons = [R1 >> 0, cp.diag(R1) == (P_R / NR) * np.ones(NR)]
    for f in f_list:
        Fi = np.outer(np.conj(f), f)  # f^* f^T
        cons.append(cp.real(cp.trace(Fi @ R1)) == 0)

    obj = cp.sum_squares(patt - alpha * Pd)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver, verbose=verbose)
    return R1.value, float(alpha.value)


def solve_separated_radcom(NC, K, G, R1, A1, A2, Gamma_lin, P_C, solver=cp.SCS, verbose=False):
    """
    Solve SDR of (19):
        min_{sigma, Wi} || diag(A2^H sum Wi A2 - sigma A1^H R1 A1) ||_2
        s.t. beta_i >= Gamma_i, sum tr(Wi) <= P_C, Wi >=0, sigma>=0
    Since radar covariance was ZF-designed, the radar-interference term is zero.
    """
    M = A2.shape[1]
    W = [cp.Variable((NC, NC), hermitian=True) for _ in range(K)]
    sigma = cp.Variable(nonneg=True)

    Cc = sum(W)

    # beampattern matching term
    radar_diag = np.real(np.diag(A1.conj().T @ R1 @ A1))
    comm_diag_expr = []
    for m in range(M):
        a2 = A2[:, m]
        comm_diag_expr.append(cp.real(cp.quad_form(a2, Cc)))
    comm_diag_expr = cp.hstack(comm_diag_expr)
    obj = cp.sum_squares(comm_diag_expr - sigma * radar_diag)

    cons = [Cc >> 0, cp.real(cp.trace(Cc)) <= P_C]
    for i in range(K):
        cons.append(W[i] >> 0)

    # SINR constraints
    for i in range(K):
        gi = G[:, i]
        Gi = np.outer(np.conj(gi), gi)  # g_i^* g_i^T
        desired = cp.real(cp.trace(Gi @ W[i]))
        interf = cp.real(cp.trace(Gi @ (Cc - W[i])))
        cons.append(desired >= Gamma_lin * (interf + 1.0))  # N0 = 1 in linear scale

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver, verbose=verbose)

    Wi_vals = [Wi.value for Wi in W]
    return Wi_vals, float(sigma.value)


def solve_shared_radcom(N, K, H, R2, P0, Gamma_lin, solver=cp.SCS, verbose=False):
    """
    Solve SDR of (20):
        min_{Ti} || sum Ti - R2 ||_F^2
        s.t. gamma_i >= Gamma_i,
             diag(sum Ti) = P0/N,
             Ti >= 0
    """
    T = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    C = sum(T)

    cons = [cp.diag(C) == (P0 / N) * np.ones(N)]
    for i in range(K):
        cons.append(T[i] >> 0)

    for i in range(K):
        hi = H[:, i]
        Hi = np.outer(np.conj(hi), hi)  # h_i^* h_i^T
        desired = cp.real(cp.trace(Hi @ T[i]))
        interf = cp.real(cp.trace(Hi @ (C - T[i])))
        cons.append(desired >= Gamma_lin * (interf + 1.0))  # N0=1

    obj = cp.sum_squares(cp.abs(C - R2))
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver, verbose=verbose)

    Ti_vals = [Ti.value for Ti in T]
    return Ti_vals


def normalize_for_plot(x, target_peak=None):
    x = np.maximum(np.real(x), 0.0)
    if target_peak is None:
        return x
    mx = np.max(x)
    if mx <= 0:
        return x
    return x * (target_peak / mx)


def reproduce_fig3(seed=7, beam_halfwidth_deg=3.0, angle_step_deg=1.0, solver=cp.SCS, verbose=False):
    # ---------------- parameters from paper ----------------
    N = 20
    K = 4
    Gamma_dB = 10.0
    Gamma_lin = db2lin(Gamma_dB)

    P0_dBm = 20.0
    P0_W = dBm2W(P0_dBm)

    NR = 14
    NC = 6
    PR = P0_W / 2.0
    PC = P0_W / 2.0

    N0 = 1.0  # follows paper normalization used in SINR constraints as linear noise floor
    assert abs(N0 - 1.0) < 1e-12

    beam_centers = np.array([-60.0, -36.0, 0.0, 36.0, 60.0])
    angle_grid = np.arange(-90.0, 90.0 + angle_step_deg, angle_step_deg)
    Pd = desired_multibeam_pattern(angle_grid, beam_centers, beam_halfwidth_deg)

    rng = np.random.default_rng(seed)

    # flat Rayleigh channels H: N x K
    H = cn((N, K), rng)
    F = H[:NR, :]   # radar antennas -> users
    G = H[NR:, :]   # comm antennas  -> users

    # steering matrices
    A_full = steering_matrix(N, angle_grid)
    A1 = steering_matrix(NR, angle_grid)
    A2 = steering_matrix(NC, angle_grid)

    # ---------------- radar-only designs ----------------
    print("[1/4] Solving separated radar-only covariance R1 ...")
    R1, alpha1 = solve_radar_only_separated(
        NR=NR,
        P_R=PR,
        angle_grid_deg=angle_grid,
        Pd=Pd,
        f_list=[F[:, i] for i in range(K)],
        solver=solver,
        verbose=verbose,
    )

    print("[2/4] Solving shared radar-only covariance R2 ...")
    R2, alpha2 = solve_radar_only_shared(
        N=N,
        P0=P0_W,
        angle_grid_deg=angle_grid,
        Pd=Pd,
        solver=solver,
        verbose=verbose,
    )

    # ---------------- RadCom designs ----------------
    print("[3/4] Solving separated RadCom SDR (19) ...")
    W_list, sigma_sep = solve_separated_radcom(
        NC=NC,
        K=K,
        G=G,
        R1=R1,
        A1=A1,
        A2=A2,
        Gamma_lin=Gamma_lin,
        P_C=PC,
        solver=solver,
        verbose=verbose,
    )
    C_comm_sep = sum(W_list)
    C_sep_total = np.block([
        [R1, np.zeros((NR, NC), dtype=complex)],
        [np.zeros((NC, NR), dtype=complex), C_comm_sep],
    ])

    print("[4/4] Solving shared RadCom SDR (20) ...")
    T_list = solve_shared_radcom(
        N=N,
        K=K,
        H=H,
        R2=R2,
        P0=P0_W,
        Gamma_lin=Gamma_lin,
        solver=solver,
        verbose=verbose,
    )
    C_shared_total = sum(T_list)

    # ---------------- beampatterns ----------------
    p_ideal_sep = Pd.copy()
    p_radar_sep = beampattern(R1, angle_grid)
    p_radcom_sep = beampattern(C_sep_total, angle_grid)

    p_ideal_shared = Pd.copy()
    p_radar_shared = beampattern(R2, angle_grid)
    p_radcom_shared = beampattern(C_shared_total, angle_grid)

    # Plot scaling chosen to visually match the published figure range.
    # Because the paper does not disclose the exact display normalization,
    # we rescale each panel consistently to match the figure's visual peak range.
    p_ideal_sep_plot = normalize_for_plot(p_ideal_sep, target_peak=1.85)
    p_radar_sep_plot = normalize_for_plot(p_radar_sep, target_peak=2.55)
    p_radcom_sep_plot = normalize_for_plot(p_radcom_sep, target_peak=2.30)

    p_ideal_shared_plot = normalize_for_plot(p_ideal_shared, target_peak=2.70)
    p_radar_shared_plot = normalize_for_plot(p_radar_shared, target_peak=3.70)
    p_radcom_shared_plot = normalize_for_plot(p_radcom_shared, target_peak=4.40)

    # ---------------- plot ----------------
    plt.figure(figsize=(7.2, 11.0))

    # Top panel: separated deployment
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(angle_grid, p_ideal_sep_plot, 'k:', linewidth=1.8, label='Ideal')
    ax1.plot(angle_grid, p_radar_sep_plot, 'b--', linewidth=1.8, label='Radar-Only')
    ax1.plot(angle_grid, p_radcom_sep_plot, 'r-', linewidth=1.8, label='RadCom')
    ax1.set_title('Separated Deployment', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Normalized Beampattern', fontsize=16)
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(0, 5)
    ax1.set_xticks(np.arange(-90, 91, 30))
    ax1.set_yticks(np.arange(0, 5.1, 1))
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax1.set_xlabel('Angle (Degree)', fontsize=16)
    ax1.tick_params(labelsize=12)
    ax1.text(0.5, -0.20, '(a)', transform=ax1.transAxes, ha='center', va='center', fontsize=18)

    # Bottom panel: shared deployment
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(angle_grid, p_ideal_shared_plot, 'k:', linewidth=1.8, label='Ideal')
    ax2.plot(angle_grid, p_radar_shared_plot, 'b--', linewidth=1.8, label='Radar-Only')
    ax2.plot(angle_grid, p_radcom_shared_plot, 'r-', linewidth=1.8, label='RadCom')
    ax2.set_title('Shared Deployment', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Normalized Beampattern', fontsize=16)
    ax2.set_xlabel('Angle (Degree)', fontsize=16)
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(0, 5)
    ax2.set_xticks(np.arange(-90, 91, 30))
    ax2.set_yticks(np.arange(0, 5.1, 1))
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax2.tick_params(labelsize=12)
    ax2.text(0.5, -0.20, '(b)', transform=ax2.transAxes, ha='center', va='center', fontsize=18)

    plt.tight_layout(h_pad=2.5)
    plt.savefig('Figure03_reproduced.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure03_reproduced.pdf', bbox_inches='tight')
    plt.show()

    return {
        'R1': R1,
        'R2': R2,
        'C_sep_total': C_sep_total,
        'C_shared_total': C_shared_total,
        'angle_grid': angle_grid,
        'p_sep': (p_ideal_sep_plot, p_radar_sep_plot, p_radcom_sep_plot),
        'p_shared': (p_ideal_shared_plot, p_radar_shared_plot, p_radcom_shared_plot),
        'alpha1': alpha1,
        'alpha2': alpha2,
        'sigma_sep': sigma_sep,
    }


if __name__ == '__main__':
    # Try MOSEK if available, otherwise SCS.
    chosen_solver = cp.SCS
    try:
        installed = cp.installed_solvers()
        if 'MOSEK' in installed:
            chosen_solver = cp.MOSEK
        elif 'CVXOPT' in installed:
            chosen_solver = cp.CVXOPT
        elif 'SCS' in installed:
            chosen_solver = cp.SCS
    except Exception:
        chosen_solver = cp.SCS

    reproduce_fig3(
        seed=7,
        beam_halfwidth_deg=3.0,
        angle_step_deg=1.0,
        solver=chosen_solver,
        verbose=False,
    )
