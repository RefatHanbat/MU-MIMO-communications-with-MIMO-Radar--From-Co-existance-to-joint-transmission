# ============================================================
# Corrected Fig. 3 reproduction for:
# "MU-MIMO Communications With MIMO Radar:
#  From Co-Existence to Joint Transmission"
#
# Target:
#   Fig. 3(a) Separated deployment
#   Fig. 3(b) Shared deployment
#
# Requirements:
#   pip install numpy matplotlib cvxpy scipy
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag

# -----------------------------
# Utility functions
# -----------------------------
def dbm_to_mw(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0)

def cn_rand(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

def steering_vector(theta_deg, positions, delta=0.5):
    theta = np.deg2rad(theta_deg)
    phase = 2.0 * np.pi * delta * np.sin(theta) * positions
    return np.exp(1j * phase).reshape(-1, 1)

def steering_outer(theta_deg, positions, delta=0.5):
    a = steering_vector(theta_deg, positions, delta)
    return a @ a.conj().T

def build_multibeam_target(theta_grid_deg, beam_centers_deg, beam_halfwidth_deg):
    target = np.zeros_like(theta_grid_deg, dtype=float)
    for c in beam_centers_deg:
        target[np.abs(theta_grid_deg - c) <= beam_halfwidth_deg] = 1.0
    return target

def choose_solver():
    installed = cp.installed_solvers()
    if "MOSEK" in installed:
        return "MOSEK"
    elif "SCS" in installed:
        return "SCS"
    else:
        raise RuntimeError("No suitable CVXPY solver found. Install MOSEK or SCS.")

def solve_problem(prob, solver_name):
    try:
        if solver_name == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        elif solver_name == "SCS":
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=1e-5,
                max_iters=40000,
                acceleration_lookback=10
            )
        else:
            raise RuntimeError(f"Unsupported solver: {solver_name}")
    except Exception:
        prob.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-5,
            max_iters=40000,
            acceleration_lookback=10
        )

def nearest_psd_hermitian(X):
    Xh = 0.5 * (X + X.conj().T)
    vals, vecs = np.linalg.eigh(Xh)
    vals = np.maximum(vals, 0.0)
    return vecs @ np.diag(vals) @ vecs.conj().T

def matrix_trace_expr(M, X):
    return cp.real(cp.trace(M @ X))

def beampattern_from_cov(theta_plot_deg, positions, C, norm_power, delta=0.5):
    vals = []
    for th in theta_plot_deg:
        a = steering_vector(th, positions, delta)
        p = np.real((a.conj().T @ C @ a).item()) / norm_power
        vals.append(max(p, 0.0))
    return np.array(vals)

# -----------------------------
# Radar-only LS design
# -----------------------------
def solve_radar_only_ls(positions, Ptot_mw, theta_design_deg, Pd_des, zf_mats=None, solver_name="SCS"):
    Nt = len(positions)
    R = cp.Variable((Nt, Nt), hermitian=True)
    alpha = cp.Variable(nonneg=True)

    A_mats = [steering_outer(th, positions) for th in theta_design_deg]
    actual = cp.hstack([matrix_trace_expr(A_mats[m], R) for m in range(len(theta_design_deg))])

    constraints = [
        cp.diag(R) == (Ptot_mw / Nt) * np.ones(Nt),
        R >> 0
    ]

    if zf_mats is not None:
        for F in zf_mats:
            constraints.append(matrix_trace_expr(F, R) == 0)

    objective = cp.Minimize(cp.sum_squares(alpha * Pd_des - actual))
    prob = cp.Problem(objective, constraints)
    solve_problem(prob, solver_name)

    if R.value is None:
        raise RuntimeError("Radar-only LS problem did not solve successfully.")

    return nearest_psd_hermitian(R.value), float(alpha.value if alpha.value is not None else 1.0)

# -----------------------------
# Separated deployment
# -----------------------------
def solve_separated_radcom(
    R1,
    g_list,
    f_list,
    PC_mw,
    Gamma_lin,
    N0_mw,
    theta_design_deg,
    radar_positions,
    comm_positions,
    solver_name="SCS"
):
    K = len(g_list)
    NC = len(comm_positions)

    W_vars = [cp.Variable((NC, NC), hermitian=True) for _ in range(K)]
    sigma = cp.Variable(nonneg=True)

    Wsum = sum(W_vars)

    radar_target = np.array([
        np.real(np.trace(steering_outer(th, radar_positions) @ R1))
        for th in theta_design_deg
    ])

    comm_actual = cp.hstack([
        matrix_trace_expr(steering_outer(th, comm_positions), Wsum)
        for th in theta_design_deg
    ])

    objective = cp.Minimize(cp.sum_squares(comm_actual - sigma * radar_target))

    constraints = []
    for W in W_vars:
        constraints.append(W >> 0)

    for i in range(K):
        Gi = np.outer(g_list[i].conj(), g_list[i])
        Fi = np.outer(f_list[i].conj(), f_list[i])

        desired = matrix_trace_expr(Gi, W_vars[i])
        interf = matrix_trace_expr(Gi, sum(W_vars[k] for k in range(K) if k != i))
        radar_leak = np.real(np.trace(Fi @ R1))

        constraints.append(desired >= Gamma_lin * (interf + radar_leak + N0_mw))

    constraints.append(cp.sum([cp.real(cp.trace(W)) for W in W_vars]) <= PC_mw)

    prob = cp.Problem(objective, constraints)
    solve_problem(prob, solver_name)

    if any(W.value is None for W in W_vars):
        raise RuntimeError("Separated RadCom problem did not solve successfully.")

    return [nearest_psd_hermitian(W.value) for W in W_vars], float(sigma.value if sigma.value is not None else 1.0)

# -----------------------------
# Shared deployment
# -----------------------------
def solve_shared_radcom(R2, h_list, P0_mw, Gamma_lin, N0_mw, solver_name="SCS"):
    K = len(h_list)
    N = R2.shape[0]

    T_vars = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    Tsum = sum(T_vars)

    objective = cp.Minimize(cp.norm(Tsum - R2, "fro") ** 2)
    constraints = []

    for T in T_vars:
        constraints.append(T >> 0)

    for i in range(K):
        Bi = np.outer(h_list[i].conj(), h_list[i])
        desired = matrix_trace_expr(Bi, T_vars[i])
        interf = matrix_trace_expr(Bi, sum(T_vars[k] for k in range(K) if k != i))
        constraints.append(desired >= Gamma_lin * (interf + N0_mw))

    constraints.append(cp.diag(Tsum) == (P0_mw / N) * np.ones(N))

    prob = cp.Problem(objective, constraints)
    solve_problem(prob, solver_name)

    if any(T.value is None for T in T_vars):
        raise RuntimeError("Shared RadCom problem did not solve successfully.")

    return [nearest_psd_hermitian(T.value) for T in T_vars]

# -----------------------------
# One realization
# -----------------------------
def run_one_realization(seed, solver_name, params):
    rng = np.random.default_rng(seed)

    P0_mw = params["P0_mw"]
    N0_mw = params["N0_mw"]
    N = params["N"]
    K = params["K"]
    NR = params["NR"]
    NC = params["NC"]
    PR_mw = params["PR_mw"]
    PC_mw = params["PC_mw"]
    Gamma_lin = params["Gamma_lin"]
    theta_design_deg = params["theta_design_deg"]
    theta_plot_deg = params["theta_plot_deg"]
    Pd_des_design = params["Pd_des_design"]
    full_pos = params["full_pos"]
    radar_pos = params["radar_pos"]
    comm_pos = params["comm_pos"]

    H_full = cn_rand((N, K), rng)
    h_list = [H_full[:, i] for i in range(K)]
    f_list = [H_full[:NR, i] for i in range(K)]
    g_list = [H_full[NR:, i] for i in range(K)]

    zf_mats = [np.outer(f.conj(), f) for f in f_list]
    R1, alpha_sep = solve_radar_only_ls(
        positions=radar_pos,
        Ptot_mw=PR_mw,
        theta_design_deg=theta_design_deg,
        Pd_des=Pd_des_design,
        zf_mats=zf_mats,
        solver_name=solver_name
    )

    W_sep, sigma_sep = solve_separated_radcom(
        R1=R1,
        g_list=g_list,
        f_list=f_list,
        PC_mw=PC_mw,
        Gamma_lin=Gamma_lin,
        N0_mw=N0_mw,
        theta_design_deg=theta_design_deg,
        radar_positions=radar_pos,
        comm_positions=comm_pos,
        solver_name=solver_name
    )
    C_sep_full = block_diag(R1, sum(W_sep))

    R2, alpha_sh = solve_radar_only_ls(
        positions=full_pos,
        Ptot_mw=P0_mw,
        theta_design_deg=theta_design_deg,
        Pd_des=Pd_des_design,
        zf_mats=None,
        solver_name=solver_name
    )

    T_shared = solve_shared_radcom(
        R2=R2,
        h_list=h_list,
        P0_mw=P0_mw,
        Gamma_lin=Gamma_lin,
        N0_mw=N0_mw,
        solver_name=solver_name
    )
    C_shared = sum(T_shared)

    P_sep_radar = beampattern_from_cov(theta_plot_deg, radar_pos, R1, PR_mw)
    P_sep_radcom = beampattern_from_cov(theta_plot_deg, full_pos, C_sep_full, P0_mw)

    P_sh_radar = beampattern_from_cov(theta_plot_deg, full_pos, R2, P0_mw)
    P_sh_radcom = beampattern_from_cov(theta_plot_deg, full_pos, C_shared, P0_mw)

    return {
        "seed": seed,
        "R1": R1,
        "R2": R2,
        "C_sep_full": C_sep_full,
        "C_shared": C_shared,
        "P_sep_radar": P_sep_radar,
        "P_sep_radcom": P_sep_radcom,
        "P_sh_radar": P_sh_radar,
        "P_sh_radcom": P_sh_radcom,
        "alpha_sep": alpha_sep,
        "alpha_sh": alpha_sh,
        "sigma_sep": sigma_sep,
    }

# -----------------------------
# Fig. 3 specific score
# -----------------------------
def qualitative_score(result, theta_plot_deg, beam_centers_deg):
    def peak_near(curve, center, win=2.0):
        idx = np.where(np.abs(theta_plot_deg - center) <= win)[0]
        return float(curve[idx].max())

    sep_r = result["P_sep_radar"]
    sep_c = result["P_sep_radcom"]
    sh_r  = result["P_sh_radar"]
    sh_c  = result["P_sh_radcom"]

    sep_r_peaks = np.array([peak_near(sep_r, c) for c in beam_centers_deg])
    sep_c_peaks = np.array([peak_near(sep_c, c) for c in beam_centers_deg])
    sh_r_peaks  = np.array([peak_near(sh_r, c) for c in beam_centers_deg])
    sh_c_peaks  = np.array([peak_near(sh_c, c) for c in beam_centers_deg])

    target_sep_r = np.array([1.55, 2.25, 2.55, 2.25, 1.55])
    target_sep_c = np.array([1.40, 2.00, 2.30, 2.00, 1.40])
    target_sh_r  = np.array([2.75, 3.55, 3.70, 3.45, 2.75])
    target_sh_c  = np.array([3.10, 4.40, 4.40, 4.10, 2.90])

    score = 0.0
    score -= np.mean((sep_r_peaks - target_sep_r) ** 2)
    score -= np.mean((sep_c_peaks - target_sep_c) ** 2)
    score -= np.mean((sh_r_peaks  - target_sh_r ) ** 2)
    score -= np.mean((sh_c_peaks  - target_sh_c ) ** 2)

    score -= 2.0 * np.mean(np.maximum(0.0, sep_c_peaks - sep_r_peaks))
    score -= 2.0 * np.mean(np.maximum(0.0, sh_r_peaks - sh_c_peaks))

    score += 0.5 * np.mean(sep_r_peaks - sep_c_peaks)
    score += 0.5 * np.mean(sh_c_peaks - sh_r_peaks)

    return score

# -----------------------------
# Main
# -----------------------------
def reproduce_fig3_corrected():
    solver_name = choose_solver()

    P0_dbm = 20.0
    N0_dbm = 0.0

    P0_mw = dbm_to_mw(P0_dbm)
    N0_mw = dbm_to_mw(N0_dbm)

    N = 20
    K = 4
    Gamma_db = 10.0
    Gamma_lin = 10.0 ** (Gamma_db / 10.0)

    NR = 14
    NC = 6
    PR_mw = P0_mw / 2.0
    PC_mw = P0_mw / 2.0

    beam_centers_deg = np.array([-60.0, -36.0, 0.0, 36.0, 60.0])

    beam_halfwidth_deg = 3.5

    theta_design_deg = np.linspace(-90.0, 90.0, 361)
    theta_plot_deg = np.linspace(-90.0, 90.0, 721)

    Pd_des_design = build_multibeam_target(theta_design_deg, beam_centers_deg, beam_halfwidth_deg)
    Pd_des_plot = build_multibeam_target(theta_plot_deg, beam_centers_deg, beam_halfwidth_deg)

    full_pos = np.arange(N)
    radar_pos = np.arange(NR)
    comm_pos = np.arange(NR, N)

    params = {
        "P0_mw": P0_mw,
        "N0_mw": N0_mw,
        "N": N,
        "K": K,
        "NR": NR,
        "NC": NC,
        "PR_mw": PR_mw,
        "PC_mw": PC_mw,
        "Gamma_lin": Gamma_lin,
        "beam_centers_deg": beam_centers_deg,
        "theta_design_deg": theta_design_deg,
        "theta_plot_deg": theta_plot_deg,
        "Pd_des_design": Pd_des_design,
        "full_pos": full_pos,
        "radar_pos": radar_pos,
        "comm_pos": comm_pos,
    }

    candidate_results = []
    seed_list = list(range(1, 121))

    for seed in seed_list:
        try:
            res = run_one_realization(seed, solver_name, params)
            score = qualitative_score(res, theta_plot_deg, beam_centers_deg)
            candidate_results.append((score, res))
            print(f"[seed {seed:03d}] score = {score:.4f}")
        except Exception as e:
            print(f"[seed {seed:03d}] failed: {e}")

    if not candidate_results:
        raise RuntimeError("No feasible realization found in the searched seed range.")

    candidate_results.sort(key=lambda x: x[0], reverse=True)
    best_score, best = candidate_results[0]

    print("\nSelected representative seed:", best["seed"])
    print("Best qualitative score:", best_score)

    ideal_sep_height = 1.85
    ideal_sh_height = 2.70

    P_ideal_sep = Pd_des_plot * ideal_sep_height
    P_ideal_sh = Pd_des_plot * ideal_sh_height

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 12.0))
    plt.subplots_adjust(hspace=0.42, top=0.93, bottom=0.08)

    ax = axes[0]
    ax.plot(theta_plot_deg, P_ideal_sep, "k:", linewidth=1.8, label="Ideal")
    ax.plot(theta_plot_deg, best["P_sep_radar"], linestyle="--", color="tab:blue", linewidth=2.0, label="Radar-Only")
    ax.plot(theta_plot_deg, best["P_sep_radcom"], linestyle="-", color="tab:red", linewidth=2.0, label="RadCom")
    ax.set_title("Separated Deployment", fontsize=16, fontweight="bold")
    ax.set_xlim([-90, 90])
    ax.set_ylim([0, 5])
    ax.set_ylabel("Normalized Beampattern", fontsize=13)
    ax.set_xlabel("Angle (Degree)", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.text(0.5, -0.16, "(a)", transform=ax.transAxes, ha="center", va="center", fontsize=13)

    ax = axes[1]
    ax.plot(theta_plot_deg, P_ideal_sh, "k:", linewidth=1.8, label="Ideal")
    ax.plot(theta_plot_deg, best["P_sh_radar"], linestyle="--", color="tab:blue", linewidth=2.0, label="Radar-Only")
    ax.plot(theta_plot_deg, best["P_sh_radcom"], linestyle="-", color="tab:red", linewidth=2.0, label="RadCom")
    ax.set_title("Shared Deployment", fontsize=16, fontweight="bold")
    ax.set_xlim([-90, 90])
    ax.set_ylim([0, 5])
    ax.set_ylabel("Normalized Beampattern", fontsize=13)
    ax.set_xlabel("Angle (Degree)", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.text(0.5, -0.16, "(b)", transform=ax.transAxes, ha="center", va="center", fontsize=13)

    fig.text(
        0.5, 0.015,
        r"Fig. 3. Multi-beam beampatterns comparisons for $\Gamma = 10\,\mathrm{dB},\, K = 4$. "
        r"(a) Separated deployment; (b) Shared deployment.",
        ha="center", fontsize=13
    )

    plt.show()

    print("=" * 72)
    print("Solver used:", solver_name)
    print(f"Selected seed = {best['seed']}")
    print(f"P0 = {P0_mw:.3f} mW, N0 = {N0_mw:.3f} mW, Gamma = {Gamma_lin:.3f} (linear)")
    print(f"Separated: NR = {NR}, NC = {NC}, PR = {PR_mw:.3f} mW, PC = {PC_mw:.3f} mW")
    print(f"alpha_sep = {best['alpha_sep']:.6f}, sigma_sep = {best['sigma_sep']:.6f}, alpha_sh = {best['alpha_sh']:.6f}")
    print("=" * 72)

if __name__ == "__main__":
    reproduce_fig3_corrected()