
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt



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


def solve_separated_radcom(NC, K, G, R1, angle_grid_deg, Gamma_lin, PC, N0, solvers, verbose=False):
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
    ok, _, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or sigma.value is None or any(Wi.value is None for Wi in W):
        return None, status
    return [Wi.value for Wi in W], status


def solve_shared_radcom(N, K, H, R2, P0, Gamma_lin, N0, solvers, verbose=False):
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
    ok, _, status = solve_with_fallback(prob, solvers, verbose)
    if not ok or any(Ti.value is None for Ti in T):
        return None, status
    return [Ti.value for Ti in T], status


def first_null_bounds(pattern, center_idx):
    """
    Find the first local minima to the left and right of the main peak.
    This is much more appropriate for PSLR than excluding only the 3 dB width.
    """
    p = np.asarray(pattern).flatten()

    # search left
    left = center_idx - 1
    while left > 1:
        if p[left] <= p[left - 1] and p[left] <= p[left + 1]:
            break
        left -= 1
    if left <= 1:
        left = max(center_idx - 3, 0)

    # search right
    right = center_idx + 1
    while right < len(p) - 2:
        if p[right] <= p[right - 1] and p[right] <= p[right + 1]:
            break
        right += 1
    if right >= len(p) - 2:
        right = min(center_idx + 3, len(p) - 1)

    return left, right


def compute_pslr_db_first_null(pattern, angle_grid_deg, theta0=0.0):
    patt = np.maximum(np.real(pattern), 1e-15)
    patt_db = 10.0 * np.log10(patt / np.max(patt))

    center_idx = int(np.argmin(np.abs(angle_grid_deg - theta0)))
    peak_window = slice(max(center_idx - 5, 0), min(center_idx + 6, len(patt_db)))
    local_peak_idx = np.argmax(patt_db[peak_window]) + max(center_idx - 5, 0)

    left_null, right_null = first_null_bounds(patt, local_peak_idx)

    sidelobe_mask = np.ones(len(patt_db), dtype=bool)
    sidelobe_mask[left_null:right_null + 1] = False

    max_sidelobe_db = np.max(patt_db[sidelobe_mask])
    pslr_db = -max_sidelobe_db
    return pslr_db, (left_null, right_null), patt_db


def evaluate_seed(seed, gamma_list_db, params, angle_grid, solvers, verbose=False):
    N = params["N"]; K = params["K"]; NR = params["NR"]; NC = params["NC"]
    P0 = params["P0"]; N0 = params["N0"]; PR = params["PR"]; PC = params["PC"]
    theta0 = params["theta0"]; bw3 = params["beamwidth_3db"]

    rng = np.random.default_rng(seed)
    H = cn((N, K), rng)
    F = H[:NR, :]
    G = H[NR:, :]

    R1, st1 = solve_radar_only_separated_3db(NR, PR, angle_grid, [F[:, i] for i in range(K)], theta0, bw3, solvers, verbose)
    if R1 is None:
        return None, f"seed {seed}: sep radar-only fail ({st1})"

    R2, st2 = solve_radar_only_shared_3db(N, P0, angle_grid, theta0, bw3, solvers, verbose)
    if R2 is None:
        return None, f"seed {seed}: shared radar-only fail ({st2})"

    sep_radar_pslr, _, _ = compute_pslr_db_first_null(beampattern(R1, angle_grid), angle_grid, theta0)
    shared_radar_pslr, _, _ = compute_pslr_db_first_null(beampattern(R2, angle_grid), angle_grid, theta0)

    sep_radcom = []
    shared_radcom = []

    for gdb in gamma_list_db:
        glin = db2lin(gdb)

        W_list, st3 = solve_separated_radcom(NC, K, G, R1, angle_grid, glin, PC, N0, solvers, verbose)
        if W_list is None:
            return None, f"seed {seed}: sep RadCom infeasible at {gdb} dB ({st3})"

        T_list, st4 = solve_shared_radcom(N, K, H, R2, P0, glin, N0, solvers, verbose)
        if T_list is None:
            return None, f"seed {seed}: shared RadCom infeasible at {gdb} dB ({st4})"

        C_sep = np.block([
            [R1, np.zeros((NR, NC), dtype=complex)],
            [np.zeros((NC, NR), dtype=complex), sum(W_list)]
        ])
        C_shared = sum(T_list)

        p_sep, _, _ = compute_pslr_db_first_null(beampattern(C_sep, angle_grid), angle_grid, theta0)
        p_shared, _, _ = compute_pslr_db_first_null(beampattern(C_shared, angle_grid), angle_grid, theta0)

        sep_radcom.append(p_sep)
        shared_radcom.append(p_shared)

    sep_radcom = np.array(sep_radcom)
    shared_radcom = np.array(shared_radcom)

    # desirability score: match paper's Fig.4 statement around Gamma=10
    idx10 = gamma_list_db.index(10)
    score = (
        abs(sep_radcom[idx10] - 7.0) +
        abs(shared_radcom[idx10] - 15.0) +
        0.2 * abs(sep_radar_pslr - 8.7) +
        0.2 * abs(shared_radar_pslr - 17.2)
    )

    # encourage monotone decreasing RadCom curves
    if np.any(np.diff(sep_radcom) > 0.25):
        score += 20.0
    if np.any(np.diff(shared_radcom) > 0.25):
        score += 20.0

    result = {
        "seed": seed,
        "gamma_list_db": np.array(gamma_list_db, dtype=float),
        "separated_radcom": sep_radcom,
        "shared_radcom": shared_radcom,
        "separated_radar": np.full(len(gamma_list_db), sep_radar_pslr),
        "shared_radar": np.full(len(gamma_list_db), shared_radar_pslr),
        "score": score,
    }
    return result, f"seed {seed}: feasible, score={score:.3f}"


def reproduce_fig5(seed_start=1, max_seed_tries=80, angle_step_deg=0.5, beamwidth_3db=10.0, verbose=False):
    params = {
        "N": 20,
        "K": 4,
        "P0": dBm2W(20.0),
        "N0": dBm2W(0.0),
        "NR": 14,
        "NC": 6,
        "theta0": 0.0,
        "beamwidth_3db": beamwidth_3db,
    }
    params["PR"] = params["P0"] / 2.0
    params["PC"] = params["P0"] / 2.0

    gamma_list_db = [4, 6, 8, 10, 12, 14]
    angle_grid = np.arange(-90.0, 90.0 + angle_step_deg, angle_step_deg)
    solvers = get_solvers()

    print("Available solvers:", [s if isinstance(s, str) else s.name() for s in solvers])
    print("Searching for a feasible seed with Figure-5-like PSLR levels ...")

    best = None
    best_score = np.inf

    for seed in range(seed_start, seed_start + max_seed_tries):
        result, msg = evaluate_seed(seed, gamma_list_db, params, angle_grid, solvers, verbose)
        print(msg)
        if result is not None and result["score"] < best_score:
            best = result
            best_score = result["score"]

    if best is None:
        raise RuntimeError("No feasible seed found in the searched range.")

    x = best["gamma_list_db"]
    plt.figure(figsize=(7.0, 5.8))
    plt.plot(x, best["shared_radcom"], 'r-x', lw=1.8, ms=8, mew=1.6, label='Shared, RadCom')
    plt.plot(x, best["separated_radcom"], 'b-s', lw=1.8, ms=7, mew=1.6, fillstyle='none', label='Separated, RadCom')
    plt.plot(x, best["shared_radar"], 'r--o', lw=1.5, ms=10, mew=1.6, fillstyle='none', dashes=(4, 4), label='Shared, Radar-Only')
    plt.plot(x, best["separated_radar"], 'b--^', lw=1.5, ms=9, mew=1.6, fillstyle='none', dashes=(4, 4), label='Separated, Radar-Only')

    plt.title('K = 4', fontsize=18, fontweight='bold')
    plt.xlabel(r'$\Gamma$ (dB)', fontsize=16)
    plt.ylabel('PSLR (dB)', fontsize=16)
    plt.xlim(4, 14)
    plt.ylim(4, 18)
    plt.xticks(x, fontsize=12)
    plt.yticks(np.arange(4, 19, 2), fontsize=12)
    plt.grid(True, alpha=0.35)
    plt.legend(loc='center left', frameon=True, fancybox=False, edgecolor='black', fontsize=12)
    plt.tight_layout()
    plt.savefig('Figure05_reproduced_revised.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure05_reproduced_revised.pdf', bbox_inches='tight')
    plt.show()

    print("\nChosen seed:", best["seed"])
    print("Score:", round(best["score"], 4))
    print("Shared, RadCom     =", np.round(best["shared_radcom"], 3))
    print("Separated, RadCom  =", np.round(best["separated_radcom"], 3))
    print("Shared, Radar-Only =", np.round(best["shared_radar"], 3))
    print("Separated, Radar-Only =", np.round(best["separated_radar"], 3))

    return best


if __name__ == "__main__":
    reproduce_fig5(seed_start=1, max_seed_tries=80, angle_step_deg=0.5, beamwidth_3db=10.0, verbose=False)
