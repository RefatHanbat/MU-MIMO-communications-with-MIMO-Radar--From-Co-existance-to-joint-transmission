import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


# ============================================================
# Fig. 7 reproduction for
# Fan Liu et al., "MU-MIMO Communications With MIMO Radar:
# From Co-Existence to Joint Transmission," IEEE TWC, 2018.
# ============================================================
# This script follows the paper's shared-deployment model and
# implements the main equations used for Fig. 7:
#   - Radar-only 3 dB beampattern design: Eq. (10)
#   - Shared constrained optimization: Eq. (20)
#   - Sum-square SINR penalty: Eqs. (23), (24), (31), (32)
#   - Max SINR penalty (log-sum-exp smoothing):
#       Eqs. (27), (40), (41)
#   - Hypersphere / oblique-manifold RCG updates:
#       Eqs. (39), (48) and Eqs. (60)-(65)
#
# Notes:
# 1) The paper uses CVX. Here CVXPY is used for the radar-only
#    SDP and the constrained SDR problem.
# 2) The weighted problems are solved by custom Riemannian
#    conjugate-gradient routines.
# 3) Exact point-by-point overlap with the published figure may
#    still vary because the paper does not disclose the original
#    random seed, the exact Monte Carlo count, nor the exact
#    angular grid density used in every figure.
# ============================================================


@dataclass
class SimConfig:
    # Paper-level defaults from Section VI
    P0_dBm: float = 20.0
    N: int = 20
    N0_dBm: float = 0.0
    d_over_lambda: float = 0.5

    # Fig. 7 setting
    K: int = 10
    theta0_deg: float = 0.0
    beamwidth_3dB_deg: float = 10.0
    gamma_req_dB_list: Tuple[float, ...] = (5.8, 6.2, 6.6, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0)

    # Table II weighting vectors
    rho_total_sumsq: Tuple[float, float] = (10.0, 1.0)
    rho_total_max: Tuple[float, float] = (10.0, 1.0)
    rho_perant_sumsq: Tuple[float, float] = (3.0, 1.0)
    rho_perant_max: Tuple[float, float] = (1.0, 2.0)

    # Monte Carlo / numerical controls
    fast_mode: bool = True
    MC_fast: int = 12
    MC_full: int = 60
    angle_grid_deg: Tuple[float, float, float] = (-90.0, 90.0, 1.0)   # beampattern grid
    plot_grid_deg: Tuple[float, float, float] = (-90.0, 90.0, 0.25)   # for PSLR evaluation
    eps_lse: float = 0.05
    rcg_tol: float = 1e-6
    rcg_max_iters: int = 250
    armijo_c1: float = 1e-4
    armijo_beta: float = 0.5
    armijo_max_backtracks: int = 25
    sdr_solver_preference: Tuple[str, ...] = ("MOSEK", "CVXOPT", "SCS")
    verbose_solver: bool = False
    seed: int = 7

    @property
    def MC(self) -> int:
        return self.MC_fast if self.fast_mode else self.MC_full


# -----------------------------
# Basic helpers
# -----------------------------
def dBm_to_watt(p_dBm: float) -> float:
    return 10.0 ** ((p_dBm - 30.0) / 10.0)


def cn_randn(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def herm(A: np.ndarray) -> np.ndarray:
    return A.conj().T


def steering_vector(theta_deg: float, N: int, d_over_lambda: float = 0.5) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    n = np.arange(N)
    return np.exp(1j * 2.0 * np.pi * d_over_lambda * n * np.sin(theta))[:, None]


def steering_outer(theta_deg: float, N: int, d_over_lambda: float = 0.5) -> np.ndarray:
    a = steering_vector(theta_deg, N, d_over_lambda)
    return a @ herm(a)


def choose_cvxpy_solver(preference: Tuple[str, ...]) -> str:
    installed = set(cp.installed_solvers())
    for name in preference:
        if name in installed:
            return name
    raise RuntimeError(
        f"None of the requested solvers {preference} is installed. Installed: {sorted(installed)}"
    )


def beampattern(C: np.ndarray, angle_deg_vec: np.ndarray, N: int, d_over_lambda: float) -> np.ndarray:
    vals = []
    for th in angle_deg_vec:
        a = steering_vector(th, N, d_over_lambda)
        vals.append(np.real((herm(a) @ C @ a).item()))
    return np.array(vals)


def _first_local_minimum_bounds(P: np.ndarray, angle_deg_vec: np.ndarray, peak_idx: int, fallback_half_width_deg: float = 10.0) -> Tuple[float, float]:
    """Estimate mainlobe bounds by searching the first local minima around the dominant peak.
    This is more appropriate for PSLR than using the 3 dB points directly, because otherwise
    the beampattern skirt right outside the 3 dB width is incorrectly counted as sidelobe.
    """
    n = len(P)

    # Left search
    left_idx = peak_idx
    for i in range(peak_idx - 1, 1, -1):
        if P[i - 1] >= P[i] and P[i + 1] >= P[i]:
            left_idx = i
            break
    else:
        left_angle = angle_deg_vec[peak_idx] - fallback_half_width_deg
        left_idx = int(np.argmin(np.abs(angle_deg_vec - left_angle)))

    # Right search
    right_idx = peak_idx
    for i in range(peak_idx + 1, n - 2):
        if P[i - 1] >= P[i] and P[i + 1] >= P[i]:
            right_idx = i
            break
    else:
        right_angle = angle_deg_vec[peak_idx] + fallback_half_width_deg
        right_idx = int(np.argmin(np.abs(angle_deg_vec - right_angle)))

    left_idx = max(0, min(left_idx, peak_idx))
    right_idx = min(n - 1, max(right_idx, peak_idx))
    return angle_deg_vec[left_idx], angle_deg_vec[right_idx]


def pslr_db(C: np.ndarray, angle_deg_vec: np.ndarray, main_region: Tuple[float, float], N: int, d_over_lambda: float) -> float:
    P = np.maximum(beampattern(C, angle_deg_vec, N, d_over_lambda), 1e-15)
    peak_idx = int(np.argmax(P))
    est_left, est_right = _first_local_minimum_bounds(
        P,
        angle_deg_vec,
        peak_idx,
        fallback_half_width_deg=max(8.0, 1.8 * 0.5 * (main_region[1] - main_region[0])),
    )

    # Use the wider region between user-provided 3 dB window and first-minimum window.
    left = min(main_region[0], est_left)
    right = max(main_region[1], est_right)
    main_mask = (angle_deg_vec >= left) & (angle_deg_vec <= right)
    side_mask = ~main_mask

    p_main = float(np.max(P[main_mask]))
    p_side = float(np.max(P[side_mask]))
    return 10.0 * np.log10(max(p_main, 1e-15) / max(p_side, 1e-15))


def avg_sinr_db_from_T(T: np.ndarray, H: np.ndarray, N0: float) -> float:
    # Eq. (7), averaged over users in dB
    K = H.shape[1]
    gammas = []
    for i in range(K):
        h = H[:, [i]]
        num = np.abs((h.T @ T[:, [i]]).item()) ** 2
        den = 0.0
        for k in range(K):
            if k != i:
                den += np.abs((h.T @ T[:, [k]]).item()) ** 2
        gammas.append(num / max(den + N0, 1e-12))
    gammas = np.array(gammas)
    return float(np.mean(10.0 * np.log10(np.maximum(gammas, 1e-12))))


def avg_sinr_db_from_covs(Tcovs: List[np.ndarray], H: np.ndarray, N0: float) -> float:
    # Eq. (7) in covariance form, useful for SDR solutions.
    C = sum(Tcovs)
    K = len(Tcovs)
    vals = []
    for i in range(K):
        h = H[:, [i]]
        B = np.conj(h) @ h.T
        sig = np.real(np.trace(B @ Tcovs[i]))
        interf = np.real(np.trace(B @ (C - Tcovs[i]))) + N0
        vals.append(sig / max(interf, 1e-12))
    vals = np.array(vals)
    return float(np.mean(10.0 * np.log10(np.maximum(vals, 1e-12))))


# -----------------------------
# Radar-only reference: Eq. (10)
# -----------------------------
def solve_radar_reference_3db(cfg: SimConfig, total_constraint: bool) -> np.ndarray:
    N = cfg.N
    P0 = dBm_to_watt(cfg.P0_dBm)
    theta0 = cfg.theta0_deg
    theta1 = theta0 - cfg.beamwidth_3dB_deg / 2.0
    theta2 = theta0 + cfg.beamwidth_3dB_deg / 2.0
    angle_grid = np.arange(*cfg.angle_grid_deg)
    sidelobe_angles = [th for th in angle_grid if (th < theta1) or (th > theta2)]

    A0 = steering_outer(theta0, N, cfg.d_over_lambda)
    A1 = steering_outer(theta1, N, cfg.d_over_lambda)
    A2 = steering_outer(theta2, N, cfg.d_over_lambda)

    R = cp.Variable((N, N), hermitian=True)
    t = cp.Variable(nonneg=True)
    constraints = [R >> 0]

    main_expr = cp.real(cp.trace(A0 @ R))
    for th in sidelobe_angles:
        Ath = steering_outer(th, N, cfg.d_over_lambda)
        constraints += [main_expr - cp.real(cp.trace(Ath @ R)) >= t]

    constraints += [cp.real(cp.trace(A1 @ R)) == 0.5 * main_expr]
    constraints += [cp.real(cp.trace(A2 @ R)) == 0.5 * main_expr]

    if total_constraint:
        constraints += [cp.real(cp.trace(R)) == P0]
    else:
        constraints += [cp.diag(R) == (P0 / N) * np.ones(N)]

    prob = cp.Problem(cp.Minimize(-t), constraints)
    solver = choose_cvxpy_solver(cfg.sdr_solver_preference)
    prob.solve(solver=solver, verbose=cfg.verbose_solver)

    if R.value is None:
        raise RuntimeError(f"Radar reference SDP failed under {'total' if total_constraint else 'per-antenna'} constraint.")

    Rv = 0.5 * (R.value + herm(R.value))
    evals, evecs = np.linalg.eigh(Rv)
    evals = np.maximum(evals, 0.0)
    return (evecs * evals[None, :]) @ herm(evecs)


# -----------------------------
# Shared constrained SDR: Eq. (20)
# -----------------------------
def solve_shared_constrained_sdr(
    H: np.ndarray,
    R_ref: np.ndarray,
    gamma_req_lin: float,
    cfg: SimConfig,
    total_constraint: bool,
) -> List[np.ndarray]:
    N, K = H.shape
    P0 = dBm_to_watt(cfg.P0_dBm)
    N0 = dBm_to_watt(cfg.N0_dBm)

    Tvars = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    C = sum(Tvars)
    constraints = [Tk >> 0 for Tk in Tvars]

    # Eq. (20b): SINR constraints in covariance form.
    for i in range(K):
        h = H[:, [i]]
        Bi = np.conj(h) @ h.T
        sig = cp.real(cp.trace(Bi @ Tvars[i]))
        interf = cp.real(cp.trace(Bi @ (C - Tvars[i]))) + N0
        constraints += [sig >= gamma_req_lin * interf]

    if total_constraint:
        constraints += [cp.real(cp.trace(C)) == P0]
    else:
        constraints += [cp.diag(C) == (P0 / N) * np.ones(N)]

    objective = cp.Minimize(cp.sum_squares(cp.abs(C - R_ref)))
    prob = cp.Problem(objective, constraints)
    solver = choose_cvxpy_solver(cfg.sdr_solver_preference)
    prob.solve(solver=solver, verbose=cfg.verbose_solver)

    if C.value is None:
        raise RuntimeError(
            f"Constrained SDR failed under {'total' if total_constraint else 'per-antenna'} constraint for gamma={10*np.log10(gamma_req_lin):.2f} dB"
        )

    out = []
    for Tk in Tvars:
        X = 0.5 * (Tk.value + herm(Tk.value))
        evals, evecs = np.linalg.eigh(X)
        evals = np.maximum(evals, 0.0)
        out.append((evecs * evals[None, :]) @ herm(evecs))
    return out


# -----------------------------
# Paper equations for penalties and gradients
# -----------------------------
def build_B_list(H: np.ndarray) -> List[np.ndarray]:
    K = H.shape[1]
    return [np.conj(H[:, [i]]) @ H[:, [i]].T for i in range(K)]


def alphas_and_G_total(T: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    # Eqs. (36)-(37)
    N, K = T.shape
    C = T @ herm(T)
    alphas = np.zeros(K, dtype=float)
    G_list = []
    for i in range(K):
        Bi = B_list[i]
        ti = T[:, [i]]
        sig_term = np.real(np.trace(Bi @ (ti @ herm(ti))))
        total_term = np.real(np.trace(Bi @ C))
        alphas[i] = (1.0 + gamma_req_lin[i]) * sig_term - gamma_req_lin[i] * total_term

        e_i = np.zeros((K, 1), dtype=complex)
        e_i[i, 0] = 1.0
        Gi = Bi @ (((1.0 + gamma_req_lin[i]) * ti @ e_i.T) - gamma_req_lin[i] * T)
        G_list.append(Gi)
    return alphas, G_list


def objective_total_sumsq(T: np.ndarray, R_ref: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray, N0: float, rho: Tuple[float, float]) -> float:
    # Eq. (32)
    alpha, _ = alphas_and_G_total(T, B_list, gamma_req_lin)
    lam = np.sum((alpha - N0 * gamma_req_lin) ** 2)  # Eq. (23)
    return rho[0] * np.linalg.norm(T @ herm(T) - R_ref, "fro") ** 2 + rho[1] * lam


def _stable_logsumexp(z: np.ndarray) -> float:
    zmax = float(np.max(z))
    return zmax + np.log(np.sum(np.exp(z - zmax)))


def _stable_softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.maximum(np.sum(ez), 1e-15)


def objective_total_max(T: np.ndarray, R_ref: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray, rho: Tuple[float, float], eps_lse: float) -> float:
    # Eq. (41) using a numerically stable log-sum-exp smoothing of Eq. (27).
    alpha, _ = alphas_and_G_total(T, B_list, gamma_req_lin)
    z = -alpha / eps_lse
    lhat = eps_lse * _stable_logsumexp(z)
    return rho[0] * np.linalg.norm(T @ herm(T) - R_ref, "fro") ** 2 + rho[1] * lhat


def grad_total_sumsq(T: np.ndarray, R_ref: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray, rho: Tuple[float, float]) -> np.ndarray:
    # Eq. (35)
    alpha, G_list = alphas_and_G_total(T, B_list, gamma_req_lin)
    Gsum = np.zeros_like(T, dtype=complex)
    for i in range(len(B_list)):
        Gsum += alpha[i] * G_list[i]
    return 4.0 * rho[0] * (T @ herm(T) - R_ref) @ T + 4.0 * rho[1] * Gsum


def grad_total_max(T: np.ndarray, R_ref: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray, rho: Tuple[float, float], eps_lse: float) -> np.ndarray:
    # Eq. right after (41) from the paper snippet.
    alpha, G_list = alphas_and_G_total(T, B_list, gamma_req_lin)
    w = _stable_softmax(-alpha / eps_lse)
    Gsum = np.zeros_like(T, dtype=complex)
    for i in range(len(B_list)):
        Gsum += w[i] * G_list[i]
    return 4.0 * rho[0] * (T @ herm(T) - R_ref) @ T - 2.0 * rho[1] * Gsum


# -----------------------------
# Riemannian tools: hypersphere
# -----------------------------
def inner_real(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.real(np.vdot(A, B)))


def proj_hypersphere(T: np.ndarray, G: np.ndarray, radius_sq: float) -> np.ndarray:
    # Tangent projection corresponding to Eq. (39).
    coeff = inner_real(T, G) / max(radius_sq, 1e-12)
    return G - coeff * T


def retract_hypersphere(T: np.ndarray, Xi: np.ndarray, radius: float) -> np.ndarray:
    Y = T + Xi
    return radius * Y / max(np.linalg.norm(Y, "fro"), 1e-12)


def armijo_backtracking(
    x: np.ndarray,
    d: np.ndarray,
    f: Callable[[np.ndarray], float],
    grad: np.ndarray,
    retract: Callable[[np.ndarray, np.ndarray], np.ndarray],
    c1: float,
    beta: float,
    max_backtracks: int,
    alpha0: float = 1.0,
) -> float:
    f0 = f(x)
    deriv = inner_real(grad, d)
    alpha = alpha0
    for _ in range(max_backtracks):
        xn = retract(x, alpha * d)
        if f(xn) <= f0 + c1 * alpha * deriv:
            return alpha
        alpha *= beta
    return alpha


def rcg_hypersphere(
    H: np.ndarray,
    R_ref: np.ndarray,
    gamma_req_lin: np.ndarray,
    cfg: SimConfig,
    rho: Tuple[float, float],
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    P0 = dBm_to_watt(cfg.P0_dBm)
    N0 = dBm_to_watt(cfg.N0_dBm)
    radius = np.sqrt(P0)
    B_list = build_B_list(H)

    T = cn_randn((cfg.N, cfg.K), rng)
    T = radius * T / np.linalg.norm(T, "fro")

    if mode == "sumsq":
        f = lambda X: objective_total_sumsq(X, R_ref, B_list, gamma_req_lin, N0, rho)
        egrad = lambda X: grad_total_sumsq(X, R_ref, B_list, gamma_req_lin, rho)
    elif mode == "max":
        f = lambda X: objective_total_max(X, R_ref, B_list, gamma_req_lin, rho, cfg.eps_lse)
        egrad = lambda X: grad_total_max(X, R_ref, B_list, gamma_req_lin, rho, cfg.eps_lse)
    else:
        raise ValueError("mode must be 'sumsq' or 'max'")

    grad = proj_hypersphere(T, egrad(T), P0)
    d = -grad

    for _ in range(cfg.rcg_max_iters):
        if np.linalg.norm(grad, "fro") <= cfg.rcg_tol:
            break
        step = armijo_backtracking(
            T,
            d,
            f,
            grad,
            lambda X, Xi: retract_hypersphere(X, Xi, radius),
            cfg.armijo_c1,
            cfg.armijo_beta,
            cfg.armijo_max_backtracks,
        )
        T_new = retract_hypersphere(T, step * d, radius)
        grad_new = proj_hypersphere(T_new, egrad(T_new), P0)

        transported_grad = proj_hypersphere(T_new, grad, P0)
        beta_pr = inner_real(grad_new, grad_new - transported_grad) / max(inner_real(grad, grad), 1e-12)
        beta_pr = max(0.0, beta_pr)  # Polak-Ribiere+

        transported_d = proj_hypersphere(T_new, d, P0)
        d = -grad_new + beta_pr * transported_d
        T, grad = T_new, grad_new

    return T


# -----------------------------
# Riemannian tools: oblique manifold
# -----------------------------
def proj_oblique(X: np.ndarray, G: np.ndarray, col_norm_sq: float) -> np.ndarray:
    # Projection corresponding to Eq. (61).
    diag_terms = np.real(np.sum(np.conj(X) * G, axis=0)) / max(col_norm_sq, 1e-12)
    return G - X @ np.diag(diag_terms)


def retract_oblique(X: np.ndarray, Xi: np.ndarray, col_norm: float) -> np.ndarray:
    # Column-wise normalization as in Eq. (62).
    Y = X + Xi
    norms = np.linalg.norm(Y, axis=0)
    norms = np.maximum(norms, 1e-12)
    return col_norm * Y / norms[None, :]


def objective_perant_sumsq(X: np.ndarray, R_ref: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray, N0: float, rho: Tuple[float, float]) -> float:
    T = herm(X)
    return objective_total_sumsq(T, R_ref, B_list, gamma_req_lin, N0, rho)


def objective_perant_max(X: np.ndarray, R_ref: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray, rho: Tuple[float, float], eps_lse: float) -> float:
    T = herm(X)
    return objective_total_max(T, R_ref, B_list, gamma_req_lin, rho, eps_lse)


def grad_perant_sumsq(X: np.ndarray, R_ref: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray, rho: Tuple[float, float]) -> np.ndarray:
    # Eq. (58) is the Hermitian-transposed counterpart of the total-space Gi form.
    T = herm(X)
    alpha, G_list = alphas_and_G_total(T, B_list, gamma_req_lin)
    GsumH = np.zeros_like(X, dtype=complex)
    for i in range(len(B_list)):
        GsumH += alpha[i] * herm(G_list[i])
    return 4.0 * rho[0] * X @ (herm(X) @ X - R_ref) + 4.0 * rho[1] * GsumH


def grad_perant_max(X: np.ndarray, R_ref: np.ndarray, B_list: List[np.ndarray], gamma_req_lin: np.ndarray, rho: Tuple[float, float], eps_lse: float) -> np.ndarray:
    # Eq. (59)
    T = herm(X)
    alpha, G_list = alphas_and_G_total(T, B_list, gamma_req_lin)
    w = _stable_softmax(-alpha / eps_lse)
    GsumH = np.zeros_like(X, dtype=complex)
    for i in range(len(B_list)):
        GsumH += w[i] * herm(G_list[i])
    return 4.0 * rho[0] * X @ (herm(X) @ X - R_ref) - 2.0 * rho[1] * GsumH


def rcg_oblique(
    H: np.ndarray,
    R_ref: np.ndarray,
    gamma_req_lin: np.ndarray,
    cfg: SimConfig,
    rho: Tuple[float, float],
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    P0 = dBm_to_watt(cfg.P0_dBm)
    N0 = dBm_to_watt(cfg.N0_dBm)
    col_norm = np.sqrt(P0 / cfg.N)
    col_norm_sq = P0 / cfg.N
    B_list = build_B_list(H)

    X = cn_randn((cfg.K, cfg.N), rng)
    X = col_norm * X / np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-12)

    if mode == "sumsq":
        f = lambda Z: objective_perant_sumsq(Z, R_ref, B_list, gamma_req_lin, N0, rho)
        egrad = lambda Z: grad_perant_sumsq(Z, R_ref, B_list, gamma_req_lin, rho)
    elif mode == "max":
        f = lambda Z: objective_perant_max(Z, R_ref, B_list, gamma_req_lin, rho, cfg.eps_lse)
        egrad = lambda Z: grad_perant_max(Z, R_ref, B_list, gamma_req_lin, rho, cfg.eps_lse)
    else:
        raise ValueError("mode must be 'sumsq' or 'max'")

    grad = proj_oblique(X, egrad(X), col_norm_sq)
    d = -grad

    for _ in range(cfg.rcg_max_iters):
        if np.linalg.norm(grad, "fro") <= cfg.rcg_tol:
            break
        step = armijo_backtracking(
            X,
            d,
            f,
            grad,
            lambda Z, Xi: retract_oblique(Z, Xi, col_norm),
            cfg.armijo_c1,
            cfg.armijo_beta,
            cfg.armijo_max_backtracks,
        )
        X_new = retract_oblique(X, step * d, col_norm)
        grad_new = proj_oblique(X_new, egrad(X_new), col_norm_sq)

        transported_grad = proj_oblique(X_new, grad, col_norm_sq)
        beta_pr = inner_real(grad_new, grad_new - transported_grad) / max(inner_real(grad, grad), 1e-12)
        beta_pr = max(0.0, beta_pr)

        transported_d = proj_oblique(X_new, d, col_norm_sq)
        d = -grad_new + beta_pr * transported_d
        X, grad = X_new, grad_new

    return X


# -----------------------------
# One Monte-Carlo realization
# -----------------------------
def solve_one_realization(
    cfg: SimConfig,
    R_total: np.ndarray,
    R_perant: np.ndarray,
    gamma_req_dB: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    H = cn_randn((cfg.N, cfg.K), rng)  # Section VI: i.i.d. standard complex Gaussian
    gamma_req_lin = np.full(cfg.K, 10.0 ** (gamma_req_dB / 10.0))
    N0 = dBm_to_watt(cfg.N0_dBm)

    # Constrained SDR
    Tcov_total = solve_shared_constrained_sdr(H, R_total, gamma_req_lin[0], cfg, total_constraint=True)
    Tcov_perant = solve_shared_constrained_sdr(H, R_perant, gamma_req_lin[0], cfg, total_constraint=False)

    C_total = sum(Tcov_total)
    C_perant = sum(Tcov_perant)

    # Weighted RCG
    T_total_sumsq = rcg_hypersphere(H, R_total, gamma_req_lin, cfg, cfg.rho_total_sumsq, mode="sumsq", rng=rng)
    T_total_max = rcg_hypersphere(H, R_total, gamma_req_lin, cfg, cfg.rho_total_max, mode="max", rng=rng)
    X_perant_sumsq = rcg_oblique(H, R_perant, gamma_req_lin, cfg, cfg.rho_perant_sumsq, mode="sumsq", rng=rng)
    X_perant_max = rcg_oblique(H, R_perant, gamma_req_lin, cfg, cfg.rho_perant_max, mode="max", rng=rng)

    C_total_sumsq = T_total_sumsq @ herm(T_total_sumsq)
    C_total_max = T_total_max @ herm(T_total_max)
    T_perant_sumsq = herm(X_perant_sumsq)
    T_perant_max = herm(X_perant_max)
    C_perant_sumsq = T_perant_sumsq @ herm(T_perant_sumsq)
    C_perant_max = T_perant_max @ herm(T_perant_max)

    plot_grid = np.arange(*cfg.plot_grid_deg)
    main_region = (
        cfg.theta0_deg - cfg.beamwidth_3dB_deg / 2.0,
        cfg.theta0_deg + cfg.beamwidth_3dB_deg / 2.0,
    )

    out = {
        "sdr_total_pslr": pslr_db(C_total, plot_grid, main_region, cfg.N, cfg.d_over_lambda),
        "sdr_total_sinr": avg_sinr_db_from_covs(Tcov_total, H, N0),
        "sumsq_total_pslr": pslr_db(C_total_sumsq, plot_grid, main_region, cfg.N, cfg.d_over_lambda),
        "sumsq_total_sinr": avg_sinr_db_from_T(T_total_sumsq, H, N0),
        "max_total_pslr": pslr_db(C_total_max, plot_grid, main_region, cfg.N, cfg.d_over_lambda),
        "max_total_sinr": avg_sinr_db_from_T(T_total_max, H, N0),
        "sdr_perant_pslr": pslr_db(C_perant, plot_grid, main_region, cfg.N, cfg.d_over_lambda),
        "sdr_perant_sinr": avg_sinr_db_from_covs(Tcov_perant, H, N0),
        "sumsq_perant_pslr": pslr_db(C_perant_sumsq, plot_grid, main_region, cfg.N, cfg.d_over_lambda),
        "sumsq_perant_sinr": avg_sinr_db_from_T(T_perant_sumsq, H, N0),
        "max_perant_pslr": pslr_db(C_perant_max, plot_grid, main_region, cfg.N, cfg.d_over_lambda),
        "max_perant_sinr": avg_sinr_db_from_T(T_perant_max, H, N0),
    }
    return out


# -----------------------------
# Figure 7 sweep
# -----------------------------
def reproduce_figure07(cfg: SimConfig) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    print("Building radar-only 3 dB references from Eq. (10) ...")
    R_total = solve_radar_reference_3db(cfg, total_constraint=True)
    R_perant = solve_radar_reference_3db(cfg, total_constraint=False)

    results = {
        "sdr_total": {"x": [], "y": []},
        "sumsq_total": {"x": [], "y": []},
        "max_total": {"x": [], "y": []},
        "sdr_perant": {"x": [], "y": []},
        "sumsq_perant": {"x": [], "y": []},
        "max_perant": {"x": [], "y": []},
    }

    for gamma_dB in cfg.gamma_req_dB_list:
        print(f"Gamma target = {gamma_dB:.2f} dB | MC = {cfg.MC}")
        bucket = {k: [] for k in [
            "sdr_total_pslr", "sdr_total_sinr",
            "sumsq_total_pslr", "sumsq_total_sinr",
            "max_total_pslr", "max_total_sinr",
            "sdr_perant_pslr", "sdr_perant_sinr",
            "sumsq_perant_pslr", "sumsq_perant_sinr",
            "max_perant_pslr", "max_perant_sinr",
        ]}

        for mc in range(cfg.MC):
            print(f"  realization {mc + 1:02d}/{cfg.MC}", flush=True)
            one = solve_one_realization(cfg, R_total, R_perant, gamma_dB, rng)
            for k, v in one.items():
                bucket[k].append(v)

        results["sdr_total"]["x"].append(np.mean(bucket["sdr_total_sinr"]))
        results["sdr_total"]["y"].append(np.mean(bucket["sdr_total_pslr"]))
        results["sumsq_total"]["x"].append(np.mean(bucket["sumsq_total_sinr"]))
        results["sumsq_total"]["y"].append(np.mean(bucket["sumsq_total_pslr"]))
        results["max_total"]["x"].append(np.mean(bucket["max_total_sinr"]))
        results["max_total"]["y"].append(np.mean(bucket["max_total_pslr"]))

        results["sdr_perant"]["x"].append(np.mean(bucket["sdr_perant_sinr"]))
        results["sdr_perant"]["y"].append(np.mean(bucket["sdr_perant_pslr"]))
        results["sumsq_perant"]["x"].append(np.mean(bucket["sumsq_perant_sinr"]))
        results["sumsq_perant"]["y"].append(np.mean(bucket["sumsq_perant_pslr"]))
        results["max_perant"]["x"].append(np.mean(bucket["max_perant_sinr"]))
        results["max_perant"]["y"].append(np.mean(bucket["max_perant_pslr"]))

    for key in results:
        results[key]["x"] = np.array(results[key]["x"])
        results[key]["y"] = np.array(results[key]["y"])

    return results


# -----------------------------
# Plot
# -----------------------------
def plot_figure07(results: Dict[str, Dict[str, np.ndarray]], cfg: SimConfig, save_path: str = "figure07_reproduced.png") -> None:
    plt.figure(figsize=(7.2, 5.8))

    # Solid = total
    plt.plot(results["max_total"]["x"], results["max_total"]["y"], "x-", linewidth=1.6, markersize=7, label="Max, RCG")
    plt.plot(results["sumsq_total"]["x"], results["sumsq_total"]["y"], "o-", linewidth=1.6, markersize=7, fillstyle="none", label="Sum-Squ, RCG")
    plt.plot(results["sdr_total"]["x"], results["sdr_total"]["y"], "^-", linewidth=1.6, markersize=8, fillstyle="none", label="Constrained, SDR")

    # Dashed = per-antenna
    plt.plot(results["max_perant"]["x"], results["max_perant"]["y"], "x--", linewidth=1.6, markersize=7)
    plt.plot(results["sumsq_perant"]["x"], results["sumsq_perant"]["y"], "o--", linewidth=1.6, markersize=7, fillstyle="none")
    plt.plot(results["sdr_perant"]["x"], results["sdr_perant"]["y"], "^--", linewidth=1.6, markersize=8, fillstyle="none")

    plt.text(8.45, 12.00, "Solid Lines: Total\nDashed Lines: Per-Ant", fontsize=11)
    plt.title(f"K = {cfg.K}", fontweight="bold")
    plt.xlabel("Average SINR (dB)")
    plt.ylabel("Average PSLR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", bbox_to_anchor=(0.86, 0.88), framealpha=1.0, fancybox=False)
    plt.xlim(5.0, 12.0)
    plt.ylim(9.0, 14.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()




def plot_figure07_paper_window(results: Dict[str, Dict[str, np.ndarray]], cfg: SimConfig, save_path: str = "figure07_reproduced_paper_window.png") -> None:
    plt.figure(figsize=(7.2, 5.8))
    x_mt, y_mt = _sorted_xy(results["max_total"]["x"], results["max_total"]["y"])
    x_st, y_st = _sorted_xy(results["sumsq_total"]["x"], results["sumsq_total"]["y"])
    x_ct, y_ct = _sorted_xy(results["sdr_total"]["x"], results["sdr_total"]["y"])
    x_mp, y_mp = _sorted_xy(results["max_perant"]["x"], results["max_perant"]["y"])
    x_sp, y_sp = _sorted_xy(results["sumsq_perant"]["x"], results["sumsq_perant"]["y"])
    x_cp, y_cp = _sorted_xy(results["sdr_perant"]["x"], results["sdr_perant"]["y"])

    plt.plot(x_mt, y_mt, "x-", linewidth=1.6, markersize=7, label="Max, RCG")
    plt.plot(x_st, y_st, "o-", linewidth=1.6, markersize=7, fillstyle="none", label="Sum-Squ, RCG")
    plt.plot(x_ct, y_ct, "^-", linewidth=1.6, markersize=8, fillstyle="none", label="Constrained, SDR")
    plt.plot(x_mp, y_mp, "x--", linewidth=1.6, markersize=7)
    plt.plot(x_sp, y_sp, "o--", linewidth=1.6, markersize=7, fillstyle="none")
    plt.plot(x_cp, y_cp, "^--", linewidth=1.6, markersize=8, fillstyle="none")

    plt.text(8.45, 12.00, "Solid Lines: Total\nDashed Lines: Per-Ant", fontsize=11)
    plt.title(f"K = {cfg.K}", fontweight="bold")
    plt.xlabel("Average SINR (dB)")
    plt.ylabel("Average PSLR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", bbox_to_anchor=(0.86, 0.88), framealpha=1.0, fancybox=False)
    plt.xlim(5.0, 12.0)
    plt.ylim(9.0, 14.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()


def main() -> None:
    cfg = SimConfig()
    print("Figure 07 reproduction")
    print(f"FAST_MODE = {cfg.fast_mode}, MC = {cfg.MC}")
    print(f"gamma_req_dB_list = {np.array(cfg.gamma_req_dB_list)}")
    print(f"Installed CVXPY solvers: {cp.installed_solvers()}")

    results = reproduce_figure07(cfg)
    for key, val in results.items():
        print(f"\n[{key}]")
        print("x (Average SINR dB):", np.round(val["x"], 3))
        print("y (Average PSLR dB):", np.round(val["y"], 3))

    # Save both an auto-scaled debug view and the paper-style window.
    plot_figure07(results, cfg, save_path="figure07_reproduced_autoscale.png")
    plot_figure07_paper_window(results, cfg, save_path="figure07_reproduced_paper_window.png")


if __name__ == "__main__":
    main()
