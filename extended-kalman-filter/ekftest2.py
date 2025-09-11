import numpy as np

def ekf_range_only(x0, P0, Q, R, U, dt, obs_pos, Z=None):
    """
    EKF for 3D position + velocity with range-only measurements.
    Inputs:
      x0: (6,) init state
      P0: (6,6) init covariance
      Q: (6,6) process noise
      R: (m,m) measurement noise
      U: (N,3) controls (acceleration)
      dt: time step
      obs_pos: (m,3) observer positions
      Z: (N,m) measurements (optional)
    Returns:
      xs: list of states
      Ps: list of covariances
    """
    x = np.asarray(x0).reshape(6)
    P = np.asarray(P0).reshape(6, 6)
    Q = np.asarray(Q).reshape(6, 6)
    obs_pos = np.asarray(obs_pos)
    m = obs_pos.shape[0]

    if Z is None:
        Z = np.full((U.shape[0], m), np.nan)
    else:
        Z = np.asarray(Z)

    if np.ndim(R) == 0:
        R = float(R) * np.eye(m)
    else:
        R = np.asarray(R).reshape(m, m)

    I = np.eye(6)

    # motion model matrices
    F = np.block([
        [np.eye(3), dt*np.eye(3)],
        [np.zeros((3,3)), np.eye(3)]
    ])
    B = np.block([
        [0.5*dt*dt*np.eye(3)],
        [dt*np.eye(3)]
    ])

    # measurement function (range)
    def h_fun(x):
        p = x[:3]
        diff = p[None, :] - obs_pos
        r = np.linalg.norm(diff, axis=1)
        return r

    # jacobian of h
    def H_jac(x):
        p = x[:3]
        diff = p[None, :] - obs_pos
        r = np.linalg.norm(diff, axis=1)
        eps = 1e-9
        inv_r = 1.0 / np.maximum(r, eps)
        Hp = diff * inv_r[:, None]
        Hv = np.zeros((m, 3))
        H = np.hstack([Hp, Hv])
        return H

    xs = [x.copy()]
    Ps = [P.copy()]

    N = U.shape[0]
    for k in range(N):
        u = U[k].reshape(3)

        # predict step
        x = F @ x + B @ u
        P = F @ P @ F.T + Q

        z = Z[k]
        if not np.all(np.isnan(z)):
            valid = ~np.isnan(z)
            z_k = z[valid]
            H = H_jac(x)[valid, :]
            z_pred = h_fun(x)[valid]
            y = z_k - z_pred

            Rv = R[np.ix_(valid, valid)]
            S = H @ P @ H.T + Rv
            K = P @ H.T @ np.linalg.pinv(S)

            # update step
            x = x + K @ y
            P = (I - K @ H) @ P
            P = 0.5 * (P + P.T)

        xs.append(x.copy())
        Ps.append(P.copy())

    return xs, Ps


# --- test the filter ---
if __name__ == "__main__":
    np.random.seed(0)

    dt = 1.0
    N = 20

    obs_pos = np.array([
        [0.0, 0.0, 0.0],
        [2000.0, 0.0, 0.0],
        [0.0, 2000.0, 100.0],
    ])
    m = obs_pos.shape[0]

    x_true = np.array([100.0, 200.0, 50.0, 5.0, -3.0, 1.0])
    u_true = np.array([0.2, 0.1, -0.05])
    U = np.tile(u_true, (N, 1))

    sigma_r = 5.0
    R = sigma_r**2 * np.eye(m)

    X_true = [x_true.copy()]
    for k in range(N):
        F = np.block([
            [np.eye(3), dt*np.eye(3)],
            [np.zeros((3,3)), np.eye(3)]
        ])
        B = np.block([
            [0.5*dt*dt*np.eye(3)],
            [dt*np.eye(3)]
        ])
        x_true = F @ x_true + B @ u_true
        X_true.append(x_true.copy())
    X_true = np.array(X_true)

    def h(p):
        d = p[None,:] - obs_pos
        return np.linalg.norm(d, axis=1)

    Z = []
    for k in range(1, N+1):
        rng = h(X_true[k,:3]) + np.random.randn(m)*sigma_r
        Z.append(rng)
    Z = np.array(Z)

    x0 = np.array([80.0, 220.0, 40.0, 0.0, 0.0, 0.0])
    P0 = np.diag([200.0, 200.0, 200.0, 25.0, 25.0, 25.0])
    Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

    xs, Ps = ekf_range_only(x0, P0, Q, R, U, dt, obs_pos, Z)

    print("final x_hat:", xs[-1])
    print("final P diag:", np.diag(Ps[-1]))
import numpy as np
