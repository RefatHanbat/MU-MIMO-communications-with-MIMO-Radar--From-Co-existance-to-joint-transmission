
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


def steering_matrix(N, angle_grid_deg, d=0.5):

    return np.hstack([steering_vector(N, th, d) for th in angle_grid_deg])


def beampattern(C, angle_grid_deg):

    vals = []

    N = C.shape[0]

    for th in angle_grid_deg:

        a = steering_vector(N, th)

        vals.append(np.real((a.conj().T @ C @ a).item()))

    return np.array(vals)


def solve_with_fallback(prob, solvers, verbose=False):

    last_exc = None

    for solver in solvers:

        try:

            prob.solve(solver=solver, verbose=verbose)

            if prob.status in ("optimal", "optimal_inaccurate"):

                return solver
            
        except Exception as e:

            last_exc = e

    raise RuntimeError(f"Optimization failed. Final status={prob.status}, last_exception={last_exc}")


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


def solve_radar_only_shared_3db(N, P0, angle_grid_deg, theta0=0.0, beamwidth_3db=10.0, solvers=None, verbose=False):

    """
    Solve paper problem (10) with Nt = N.
    Main beam center at theta0, 3dB beamwidth = beamwidth_3db.
    We set theta1 = theta0 - beamwidth_3db/2, theta2 = theta0 + beamwidth_3db/2,
    and sidelobe region Omega = all grid points excluding [theta1, theta2].
    """
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

    used = solve_with_fallback(prob, solvers, verbose)


    if R.value is None:

        raise RuntimeError(f"Shared radar-only 3dB failed. status={prob.status}, solver={used}")
    
    return R.value, float(t.value), prob.status, used


def solve_radar_only_separated_3db(NR, PR, angle_grid_deg, f_list, theta0=0.0, beamwidth_3db=10.0, solvers=None, 
                                   verbose=False):
    """
    Solve paper problem (13).
    """
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

    cons = [

        R1 >> 0,

        cp.diag(R1) == (PR / NR) * np.ones(NR),

        p1 == p0 / 2.0,

        p2 == p0 / 2.0,
    ]

    for f in f_list:

        Fi = np.outer(np.conj(f), f)

        cons.append(cp.real(cp.trace(Fi @ R1)) == 0)


    sidelobe_angles = [th for th in angle_grid_deg if (th < theta1 or th > theta2)]

    for th in sidelobe_angles:

        a = steering_vector(NR, th).flatten()

        pm = cp.real(cp.quad_form(a, R1))

        cons.append(p0 - pm >= t)


    prob = cp.Problem(cp.Minimize(-t), cons)

    used = solve_with_fallback(prob, solvers, verbose)


    if R1.value is None:

        raise RuntimeError(f"Separated radar-only 3dB failed. status={prob.status}, solver={used}")
    
    return R1.value, float(t.value), prob.status, used


def solve_separated_radcom(NC, K, G, R1, A1, A2, Gamma_lin, PC, N0, solvers, verbose=False):

    """
    SDR of (19), using the radar covariance R1 from the separated radar-only design.
    """

    W = [cp.Variable((NC, NC), hermitian=True) for _ in range(K)]

    sigma = cp.Variable(nonneg=True)

    Cc = sum(W)

    radar_diag = np.real(np.diag(A1.conj().T @ R1 @ A1))

    comm_diag = []

    for m in range(A2.shape[1]):

        a2 = A2[:, m]

        comm_diag.append(cp.real(cp.quad_form(a2, Cc)))

    comm_diag = cp.hstack(comm_diag)

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

    used = solve_with_fallback(prob, solvers, verbose)


    if sigma.value is None or any(Wi.value is None for Wi in W):

        raise RuntimeError(f"Separated RadCom failed. status={prob.status}, solver={used}")
    
    return [Wi.value for Wi in W], float(sigma.value), prob.status, used


def solve_shared_radcom(N, K, H, R2, P0, Gamma_lin, N0, solvers, verbose=False):

    """
    SDR of (20), using the radar covariance R2 from the shared radar-only design.
    """

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

    used = solve_with_fallback(prob, solvers, verbose)

    if any(Ti.value is None for Ti in T):

        raise RuntimeError(f"Shared RadCom failed. status={prob.status}, solver={used}")
    
    return [Ti.value for Ti in T], prob.status, used


def normalize_db(x):

    x = np.maximum(np.real(x), 1e-12)

    x = x / np.max(x)

    return 10.0 * np.log10(x)



def reproduce_fig4(seed=7, angle_step_deg=1.0, beamwidth_3db=10.0, verbose=False):

    # Paper parameters


    N = 20

    K = 4

    P0_dBm = 20.0

    N0_dBm = 0.0

    Gamma_dB = 10.0

    P0 = dBm2W(P0_dBm)

    N0 = dBm2W(N0_dBm)

    Gamma_lin = db2lin(Gamma_dB)


    NR = 14

    NC = 6

    PR = P0 / 2.0

    PC = P0 / 2.0


    rng = np.random.default_rng(seed)

    H = cn((N, K), rng)

    F = H[:NR, :]

    G = H[NR:, :]

    angle_grid = np.arange(-90.0, 90.0 + angle_step_deg, angle_step_deg)

    A1 = steering_matrix(NR, angle_grid)

    A2 = steering_matrix(NC, angle_grid)

    solvers = get_solvers()

    print("Available solvers:", [s if isinstance(s, str) else s.name() for s in solvers])


    print("[1/4] Solving separated radar-only 3dB design (13) ...")

    R1, t1, st1, sv1 = solve_radar_only_separated_3db(

        NR=NR, PR=PR, angle_grid_deg=angle_grid, f_list=[F[:, i] for i in range(K)],
        theta0=0.0, beamwidth_3db=beamwidth_3db, solvers=solvers, verbose=verbose

    )

    print(f"      status={st1}, solver={sv1}, t={t1:.6f}")


    print("[2/4] Solving shared radar-only 3dB design (10) ...")

    R2, t2, st2, sv2 = solve_radar_only_shared_3db(

        N=N, P0=P0, angle_grid_deg=angle_grid,

        theta0=0.0, beamwidth_3db=beamwidth_3db, solvers=solvers, verbose=verbose

    )

    print(f"      status={st2}, solver={sv2}, t={t2:.6f}")


    print("[3/4] Solving separated RadCom SDR (19) ...")

    W_list, sigma_sep, st3, sv3 = solve_separated_radcom(

        NC=NC, K=K, G=G, R1=R1, A1=A1, A2=A2,

        Gamma_lin=Gamma_lin, PC=PC, N0=N0, solvers=solvers, verbose=verbose
    )

    print(f"      status={st3}, solver={sv3}, sigma={sigma_sep:.6f}")

    Cc = sum(W_list)

    C_sep = np.block([
        [R1, np.zeros((NR, NC), dtype=complex)],
        [np.zeros((NC, NR), dtype=complex), Cc]
    ])

    print("[4/4] Solving shared RadCom SDR (20) ...")

    T_list, st4, sv4 = solve_shared_radcom(

        N=N, K=K, H=H, R2=R2, P0=P0,

        Gamma_lin=Gamma_lin, N0=N0, solvers=solvers, verbose=verbose
    )

    print(f"      status={st4}, solver={sv4}")

    C_shared = sum(T_list)


    p_sep_radar = beampattern(R1, angle_grid)

    p_sep_radcom = beampattern(C_sep, angle_grid)


    p_shared_radar = beampattern(R2, angle_grid)

    p_shared_radcom = beampattern(C_shared, angle_grid)


    y_sep_radar = normalize_db(p_sep_radar)

    y_sep_radcom = normalize_db(p_sep_radcom)

    y_shared_radar = normalize_db(p_shared_radar)

    y_shared_radcom = normalize_db(p_shared_radcom)

    plt.figure(figsize=(7.2, 11.0))


    ax1 = plt.subplot(2, 1, 1)

    ax1.plot(angle_grid, y_sep_radar, 'b--', lw=1.8, label='Radar-Only')

    ax1.plot(angle_grid, y_sep_radcom, 'r-', lw=1.8, label='RadCom')

    ax1.set_title('Separated Deployment', fontsize=18, fontweight='bold')

    ax1.set_ylabel('Normalized Beampattern (dBi)', fontsize=16)

    ax1.set_xlim(-90, 90)

    ax1.set_ylim(-20, 15)

    ax1.set_xticks(np.arange(-90, 91, 30))

    ax1.set_yticks(np.arange(-20, 16, 5))

    ax1.grid(True, alpha=0.35)

    ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    ax1.set_xlabel('Angle (Degree)', fontsize=16)

    ax1.tick_params(labelsize=12)

    ax1.text(0.5, -0.22, '(a)', transform=ax1.transAxes, ha='center', va='center', fontsize=18)


    ax2 = plt.subplot(2, 1, 2)

    ax2.plot(angle_grid, y_shared_radar, 'b--', lw=1.8, label='Radar-Only')

    ax2.plot(angle_grid, y_shared_radcom, 'r-', lw=1.8, label='RadCom')

    ax2.set_title('Shared Deployment', fontsize=18, fontweight='bold')

    ax2.set_ylabel('Normalized Beampattern (dBi)', fontsize=16)

    ax2.set_xlabel('Angle (Degree)', fontsize=16)

    ax2.set_xlim(-90, 90)

    ax2.set_ylim(-20, 15)

    ax2.set_xticks(np.arange(-90, 91, 30))

    ax2.set_yticks(np.arange(-20, 16, 5))

    ax2.grid(True, alpha=0.35)

    ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    ax2.tick_params(labelsize=12)

    ax2.text(0.5, -0.22, '(b)', transform=ax2.transAxes, ha='center', va='center', fontsize=18)


    plt.tight_layout(h_pad=3.0)

    plt.savefig('Figure04_reproduced.png', dpi=300, bbox_inches='tight')

    plt.savefig('Figure04_reproduced.pdf', bbox_inches='tight')

    plt.show()


    return {

        "R1": R1,

        "R2": R2,

        "C_sep": C_sep,

        "C_shared": C_shared,

        "sigma_sep": sigma_sep,


        "angle_grid": angle_grid,

        "y_sep_radar": y_sep_radar,

        "y_sep_radcom": y_sep_radcom,

        "y_shared_radar": y_shared_radar,

        "y_shared_radcom": y_shared_radcom,

    }


if __name__ == "__main__":
    
    reproduce_fig4(seed=7, angle_step_deg=1.0, beamwidth_3db=10.0, verbose=False)
