import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N_meas = 10
N_plot = 1000

groundtruth_targets = np.array([0.8027692908754149, 0.0, 0.0, -1.1309830924549648e-14, 0.33765564334938736, 0.0])
orbit_period = 3.225
mu = 1.215058560962404e-02
LU = 384400.0
TU = 3.751902619517228e5
x_observer = [1.0 - mu, 0.0, 0.0, 0.0, 0.0, 0.0]

t0 = 0.0
t_measurements = np.linspace(orbit_period / N_meas, orbit_period, N_meas)

def cr3bp(x, u):
    X, Y, Z, VX, VY, VZ = x
    r1 = np.sqrt((X + mu)**2 + Y**2 + Z**2)
    r2 = np.sqrt((X - 1 + mu)**2 + Y**2 + Z**2)
    ax =  2*VY + X - (1 - mu)*(X + mu)/r1**3 - mu*(X - 1 + mu)/r2**3 + u[0]
    ay = -2*VX + Y - (1 - mu)*Y/r1**3       - mu*Y/r2**3       + u[1]
    az =               - (1 - mu)*Z/r1**3   - mu*Z/r2**3       + u[2]
    return np.array([VX, VY, VZ, ax, ay, az], dtype=float)

def fstep(x, u, dt):
    def rhs(t, s):
        return cr3bp(s, u)
    sol = solve_ivp(rhs, (0.0, float(dt)), np.asarray(x, float), method="DOP853", rtol=1e-12, atol=1e-12, max_step=float(dt))
    return sol.y[:, -1]

def jacobianF(x, u, dt, eps=1e-7):
    n = x.size
    Fk = np.zeros((n, n))
    fx = fstep(x, u, dt)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        fxi = fstep(x + dx, u, dt)
        Fk[:, i] = (fxi - fx) / eps
    return Fk

def _wrap_angle(a):
    return (a + np.pi) % (2.0*np.pi) - np.pi

def meas_angles_rates(x, obspos):
    p = x[:3]
    v = x[3:6]
    out = []
    for r_obs in obspos:
        rx, ry, rz = (p - r_obs)
        vx, vy, vz = v
        rho2 = rx*rx + ry*ry
        rho  = np.sqrt(max(rho2, 1e-16))
        r2   = rho2 + rz*rz
        az = np.arctan2(ry, rx)
        el = np.arctan2(rz, rho)
        azdot  = (rx*vy - ry*vx) / max(rho2, 1e-16)
        rhodot = (rx*vx + ry*vy) / rho
        eldot  = (vz*rho - rz*rhodot) / max(r2, 1e-20)
        out.extend([az, el, azdot, eldot])
    return np.array(out, dtype=float)

def Hjac_num(x, obspos, eps=1e-7):
    h0 = meas_angles_rates(x, obspos)
    m = h0.size
    H = np.zeros((m, 6))
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = eps
        hi = meas_angles_rates(x + dx, obspos)
        hiw = hi.copy()
        h0w = h0.copy()
        for k in range(0, m, 4):
            hiw[k+0] = _wrap_angle(hi[k+0])
            hiw[k+1] = _wrap_angle(hi[k+1])
            h0w[k+0] = _wrap_angle(h0[k+0])
            h0w[k+1] = _wrap_angle(h0[k+1])
        H[:, i] = (hiw - h0w) / eps
    return H

def ekf_angles_var(x0, P0, Q, Rsingle, U_seq, dt_seq, obspos, Z):
    x = np.asarray(x0, float).reshape(6)
    P = np.asarray(P0, float).reshape(6, 6)
    Q = np.asarray(Q, float).reshape(6, 6)
    obspos = np.asarray(obspos, float)
    mstations = obspos.shape[0]
    R = np.kron(np.eye(mstations), np.asarray(Rsingle, float).reshape(4, 4))
    I = np.eye(6)
    xs = [x.copy()]
    Ps = [P.copy()]
    residuals = []
    zpreds = []
    K = len(dt_seq)
    for k in range(K):
        u = U_seq[k]
        dt = dt_seq[k]
        xprev = x.copy()
        x = fstep(xprev, u, dt)
        Fk = jacobianF(xprev, u, dt)
        P = Fk @ P @ Fk.T + Q
        z = Z[k]
        H = Hjac_num(x, obspos)
        zpred = meas_angles_rates(x, obspos)
        y = z - zpred
        for i0 in range(0, 4*mstations, 4):
            y[i0+0] = _wrap_angle(y[i0+0])
            y[i0+1] = _wrap_angle(y[i0+1])
        S = H @ P @ H.T + R
        Kk = P @ H.T @ np.linalg.pinv(S)
        x = x + Kk @ y
        P = (I - Kk @ H) @ P @ (I - Kk @ H).T + Kk @ R @ Kk.T
        P = 0.5 * (P + P.T)
        xs.append(x.copy())
        Ps.append(P.copy())
        residuals.append(y.copy())
        zpreds.append(zpred.copy())
    return np.array(xs), Ps, np.array(residuals), np.array(zpreds)

def rollout_states(t_steps, Xest, U_seq, t_plot):
    out = np.zeros((t_plot.size, 6))
    idx = 0
    for k in range(len(U_seq)):
        t0k = t_steps[k]
        t1k = t_steps[k+1]
        if k == len(U_seq) - 1:
            mask = (t_plot >= t0k) & (t_plot <= t1k)
        else:
            mask = (t_plot >= t0k) & (t_plot <  t1k)
        tseg = t_plot[mask]
        if tseg.size == 0:
            continue
        def rhs(t, s):
            return cr3bp(s, U_seq[k])
        sol = solve_ivp(rhs, (t0k, t1k), Xest[k], t_eval=tseg, method="DOP853", rtol=1e-12, atol=1e-12)
        out[idx:idx + tseg.size, :] = sol.y.T
        idx += tseg.size
    return out

np.random.seed(2)
t_all = np.concatenate([[t0], t_measurements])
dt_seq = np.diff(t_all)
U_seq = np.zeros((len(dt_seq), 3))
obspos = np.array([x_observer[:3]])

x_truth = groundtruth_targets.copy()
Xtruth_at_meas = [x_truth.copy()]
for k in range(len(dt_seq)):
    x_truth = fstep(x_truth, np.zeros(3), dt_seq[k])
    Xtruth_at_meas.append(x_truth.copy())
Xtruth_at_meas = np.array(Xtruth_at_meas)

def meas_from_posvel(p, v, obspos):
    return meas_angles_rates(np.hstack([p, v]), obspos)

sigma_az   = 1e-3
sigma_el   = 1e-3
sigma_azd  = 2e-3
sigma_eld  = 2e-3
Rsingle = np.diag([sigma_az**2, sigma_el**2, sigma_azd**2, sigma_eld**2])

Z = []
for k in range(1, len(t_all)):
    clean = meas_from_posvel(Xtruth_at_meas[k, :3], Xtruth_at_meas[k, 3:6], obspos)
    noise = np.array([np.random.randn()*sigma_az, np.random.randn()*sigma_el, np.random.randn()*sigma_azd, np.random.randn()*sigma_eld])
    m = clean + noise
    m[0] = _wrap_angle(m[0])
    m[1] = _wrap_angle(m[1])
    Z.append(m)
Z = np.array(Z)

x0est = groundtruth_targets + np.array([3e-3, -2e-3, 2e-3, 0.0, 0.0, 0.0])
P0 = np.diag([2e-2, 2e-2, 2e-2, 5e-3, 5e-3, 5e-3])
Q  = np.diag([1e-9, 1e-9, 1e-9, 5e-7, 5e-7, 5e-7])

Xest, Ps, residuals, zpreds = ekf_angles_var(x0est, P0, Q, Rsingle, U_seq, dt_seq, obspos, Z)

print("final xhat:", Xest[-1])
print("final P diag:", np.diag(Ps[-1]))

t_steps = t_all
t_plot = np.linspace(t_steps[0], t_steps[-1], N_plot)
t_seconds_plot = t_plot * TU
Vmps = (LU * 1000.0) / TU

sig_all = np.array([np.diag(Ps[k]) for k in range(len(Ps))])
sig_pos = np.sqrt(sig_all[:, 0:3])
sig_vel = np.sqrt(sig_all[:, 3:6])

Xest_plot = rollout_states(t_steps, Xest, U_seq, t_plot)
sig_pos_plot = np.vstack([np.interp(t_plot, t_steps, sig_pos[:, j]) for j in range(3)]).T
sig_vel_plot = np.vstack([np.interp(t_plot, t_steps, sig_vel[:, j]) for j in range(3)]).T

pos_est_km_plot = Xest_plot[:, 0:3] * LU
vel_est_ms_plot = Xest_plot[:, 3:6] * Vmps
sig_pos_km_plot = sig_pos_plot * LU
sig_vel_ms_plot = sig_vel_plot * Vmps

fig1, axarr = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
names = ["x km", "y km", "z km", "vx m/s", "vy m/s", "vz m/s"]
seriesX = [pos_est_km_plot[:,0], pos_est_km_plot[:,1], pos_est_km_plot[:,2],
           vel_est_ms_plot[:,0], vel_est_ms_plot[:,1], vel_est_ms_plot[:,2]]
seriesS = [sig_pos_km_plot[:,0], sig_pos_km_plot[:,1], sig_pos_km_plot[:,2],
           sig_vel_ms_plot[:,0], sig_vel_ms_plot[:,1], sig_vel_ms_plot[:,2]]

for i, ax in enumerate(axarr.ravel()):
    xhat = seriesX[i]
    sig1 = seriesS[i]
    ax.fill_between(t_seconds_plot, xhat - 3.0*sig1, xhat + 3.0*sig1, alpha=0.25)
    ax.plot(t_seconds_plot, xhat, linewidth=1.6)
    ax.set_title(names[i])
    ax.grid(True)
for ax in axarr[-1, :]:
    ax.set_xlabel("Time s")
fig1.suptitle(f"EKF Estimates ±3σ (N_meas={N_meas}, N_plot={N_plot})")
fig1.tight_layout()
fig1.savefig("ekf_estimate_bands.png", dpi=150)

fig2 = plt.figure(figsize=(8, 6))
ax3d = fig2.add_subplot(111, projection="3d")
ax3d.plot(pos_est_km_plot[:,0], pos_est_km_plot[:,1], pos_est_km_plot[:,2], label="estimate")
ax3d.set_xlabel("x km")
ax3d.set_ylabel("y km")
ax3d.set_zlabel("z km")
ax3d.set_title("Estimated Trajectory (3D)")
ax3d.legend(loc="best")
fig2.tight_layout()
fig2.savefig("ekf_estimated_traj3d.png", dpi=150)

t_meas_seconds = t_measurements * TU
fig3, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
labs = ["az (rad)", "el (rad)", "azdot (rad/s nd)", "eldot (rad/s nd)"]
for i in range(4):
    axs[i].plot(t_meas_seconds, residuals[:, i], marker="o")
    axs[i].grid(True)
    axs[i].set_ylabel(labs[i])
axs[-1].set_xlabel("Time s")
fig3.suptitle("Measurement Residuals (innovation y)")
fig3.tight_layout()
fig3.savefig("ekf_residuals.png", dpi=150)

state_table = np.column_stack([np.arange(Xest.shape[0], dtype=int), t_steps, Xest])
np.savetxt("ekf_state_history.txt",
           state_table,
           fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e"],
           header="k t x y z vx vy vz")

with open("ekf_cov_history.txt", "w") as f:
    for i, P in enumerate(Ps):
        f.write(f"# k {i}\n")
        np.savetxt(f, P, fmt="%.10e")
        f.write("\n")

plt.close('all')
print(t_seconds_plot.shape)   # N_plot
print(t_meas_seconds.shape)   # N_meas
