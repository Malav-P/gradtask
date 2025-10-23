import numpy as np

mu = 1.215058560962404e-02

def dyn_cr3bp(x, u):
    X, Y, Z, VX, VY, VZ = x
    r1 = np.sqrt((X + mu)**2 + Y**2 + Z**2)
    r2 = np.sqrt((X - 1 + mu)**2 + Y**2 + Z**2)
    ax =  2*VY + X - (1 - mu)*(X + mu)/r1**3 - mu*(X - 1 + mu)/r2**3 + u[0]
    ay = -2*VX + Y - (1 - mu)*Y/r1**3       - mu*Y/r2**3       + u[1]
    az =               - (1 - mu)*Z/r1**3   - mu*Z/r2**3       + u[2]
    return np.array([VX, VY, VZ, ax, ay, az], dtype=float)

def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def f_step(x, u, dt):
    return rk4_step(dyn_cr3bp, x, u, dt)

def jacobian_F(x, u, dt, eps=1e-6):
    n = x.size
    Fk = np.zeros((n, n))
    fx = f_step(x, u, dt)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        fxi = f_step(x + dx, u, dt)
        Fk[:, i] = (fxi - fx) / eps
    return Fk

def h_fun(x, obs_pos):
    p = x[:3]
    diff = p[None, :] - obs_pos
    return np.linalg.norm(diff, axis=1)

def H_jac(x, obs_pos):
    p = x[:3]
    diff = p[None, :] - obs_pos
    r = np.linalg.norm(diff, axis=1)
    eps = 1e-9
    inv_r = 1.0 / np.maximum(r, eps)
    Hp = diff * inv_r[:, None]
    Hv = np.zeros((obs_pos.shape[0], 3))
    return np.hstack([Hp, Hv])

def ekf_range_only(x0, P0, Q, R, U, dt, obs_pos, Z=None):
    x = np.asarray(x0, float).reshape(6)
    P = np.asarray(P0, float).reshape(6, 6)
    Q = np.asarray(Q, float).reshape(6, 6)
    obs_pos = np.asarray(obs_pos, float)
    m = obs_pos.shape[0]
    if Z is None:
        Z = np.full((U.shape[0], m), np.nan, float)
    else:
        Z = np.asarray(Z, float)
    if np.ndim(R) == 0:
        R = float(R) * np.eye(m)
    else:
        R = np.asarray(R, float).reshape(m, m)
    I = np.eye(6)
    xs = [x.copy()]
    Ps = [P.copy()]
    N = U.shape[0]
    for k in range(N):
        u = U[k].reshape(3)
        x_prev = x.copy()
        x = f_step(x_prev, u, dt)
        Fk = jacobian_F(x_prev, u, dt)
        P = Fk @ P @ Fk.T + Q
        z = Z[k]
        if not np.all(np.isnan(z)):
            valid = ~np.isnan(z)
            H = H_jac(x, obs_pos)[valid, :]
            z_pred = h_fun(x, obs_pos)[valid]
            y = z[valid] - z_pred
            Rv = R[np.ix_(valid, valid)]
            S = H @ P @ H.T + Rv
            K = P @ H.T @ np.linalg.pinv(S)
            x = x + K @ y
            P = (I - K @ H) @ P
            P = 0.5 * (P + P.T)
        xs.append(x.copy())
        Ps.append(P.copy())
    return xs, Ps

if __name__ == "__main__":
    np.random.seed(0)
    dt = 0.01
    N = 200
    obs_pos = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.25, 0.05],
    ])
    m = obs_pos.shape[0]
    x_true = np.array([0.8, 0.0, 0.0, 0.0, 0.35, 0.0])
    u_true = np.array([0.0, 0.0, 0.0])
    U = np.tile(u_true, (N, 1))
    X_true = [x_true.copy()]
    for k in range(N):
        x_true = f_step(x_true, u_true, dt)
        X_true.append(x_true.copy())
    X_true = np.array(X_true)
    def h_pos(p): return np.linalg.norm(p[None, :] - obs_pos, axis=1)
    sigma_r = 1e-3
    R = sigma_r**2 * np.eye(m)
    Z = []
    for k in range(1, N+1):
        rng = h_pos(X_true[k, :3]) + np.random.randn(m)*sigma_r
        Z.append(rng)
    Z = np.array(Z)
    x0 = X_true[0] + np.array([1e-2, -1e-2, 5e-3, 0.0, 0.0, 0.0])
    P0 = np.diag([1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])
    Q  = np.diag([1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6])
    xs, Ps = ekf_range_only(x0, P0, Q, R, U, dt, obs_pos, Z)
    print("final x_hat:", xs[-1])
    print("final P diag:", np.diag(Ps[-1]))
