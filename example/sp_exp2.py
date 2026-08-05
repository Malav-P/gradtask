# from root : python3 -m example.exp7

# note: using computed information (using jax)
# note: need to clip gradients to avoid large changes in x after one step


from blackboxphaseopt.shortest_path_problem import solve_shortest_path_time_expanded, solve_shortest_path
from blackboxphaseopt.dynamics import gen_state_history, build_taylor_cr3bp
from blackboxphaseopt.gradients import  compute_generalized_distances, compute_projected_gradients
from blackboxphaseopt.optimizers import SGD
from blackboxphaseopt.constants import Config, CR3BP_MU


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def gd_loop():
    orbit_pos_z = [9.2828906799831279E-1, -6.2106134091203751E-26,  2.9579500586038443E-1,  3.8020223639592665E-11,  8.1841658932217537E-2, -1.7711888929326446E-10]
    orbit_neg_z = [9.2828906799831279E-1, -6.2106134091203751E-26, -2.9579500586038443E-1,  3.8020223639592665E-11,  8.1841658932217537E-2,  1.7711888929326446E-10]
    initial_state = np.array([
        orbit_pos_z,
        orbit_pos_z,
        orbit_neg_z,
        orbit_neg_z,
    ])

    config = Config(
        exp_name = "sp_exp2",
        period = 2.1721303582818936,
        n_points = 215,
        max_iters = (175,),
        initial_state_observer= initial_state,
        initial_state_observer_phases=[(0.5, *np.random.uniform(0, 1, 3))]
    )
    

    ta = build_taylor_cr3bp(mu=config.mu, stm=False, batched=True)
    n_satellites = len(config.initial_state_observer_phases[0])
    earth_x = np.array([-CR3BP_MU, 0, 0, 0, 0, 0])

    earth_expanded = np.tile(earth_x, (config.n_points, 1))        # (T, 6)
    earth_expanded = earth_expanded[None, :, :]    # (1, T, 6)

    L2 = np.array([1 - CR3BP_MU + (CR3BP_MU / 3) ** (1/3), 0., 0.])
    sigma_L2 = 0.05   # spread of the fuzzy ball around L2 (DU)

    for p, max_iter in zip(config.initial_state_observer_phases, config.max_iters):

        grad_history = np.empty(shape=(max_iter, n_satellites))
        obj_history = np.empty(shape=(max_iter,))
        phase_history= np.empty(shape=(max_iter, n_satellites))

        optimizer = SGD(tuple(p[1:]), modulo=1, momentum=0.0, lr=1, noise=0, clip_grad_norm=10, cosine_anneal=True, t_max=max_iter, eta_min=1e-2)

        for n_iter in range(max_iter):

            _, states_x = gen_state_history(ta=ta,
                                    initial_state=config.initial_state_observer,
                                    time=config.period,
                                    n_points=config.n_points,
                                    phase=(0.5, *optimizer.parameters))

            states_target = np.zeros((1, config.n_points, 6))
            states_target[0, :, :3] = np.random.randn(config.n_points, 3) * sigma_L2 + L2

            states_x = np.concatenate([states_x, earth_expanded, states_target], axis=0)
            dist, grad = compute_generalized_distances(states_x=states_x, states_y=states_x, Q=np.array([1, 1, 1, 0, 0, 0]))

        

            x, _, obj = solve_shortest_path_time_expanded(dist, n_satellites, n_satellites + 1)
            obj /= config.n_points
            obj_history[n_iter] = obj

            masked_gradients = grad * x[..., np.newaxis]  # shape: (T, N, N, state_dim)

            g = np.empty(shape=(n_satellites, config.n_points, 6))
            for i in range(n_satellites):
                g[i] = np.sum(masked_gradients[:, i, :, :], axis=-2) - np.sum(masked_gradients[:, :, i, :], axis=-2)


            proj_g = compute_projected_gradients(g, states=states_x[:n_satellites], reduction="mean", weights=None)

            grad_history[n_iter] = proj_g

            optimizer.step(proj_g[1:])

            phase = (0.5, *optimizer.parameters)
            phase_history[n_iter] = phase

            grad_str   = "[" + ", ".join(f"{v:.4f}" for v in proj_g) + "]"
            phase_str  = "[" + ", ".join(f"{v:.4f}" for v in phase)  + "]"
            print(f"Gradient: {grad_str}  Objective: {obj:.4f}  New Phases: {phase_str}")

    return phase, phase_history



def plot_phase_evolution(phase_history):
    orbit_pos_z = [9.2828906799831279E-1, -6.2106134091203751E-26,  2.9579500586038443E-1,  3.8020223639592665E-11,  8.1841658932217537E-2, -1.7711888929326446E-10]
    orbit_neg_z = [9.2828906799831279E-1, -6.2106134091203751E-26, -2.9579500586038443E-1,  3.8020223639592665E-11,  8.1841658932217537E-2,  1.7711888929326446E-10]
    initial_state = np.array([orbit_pos_z, orbit_pos_z, orbit_neg_z, orbit_neg_z])

    period   = 2.1721303582818936
    n_points = 215
    ta = build_taylor_cr3bp(mu=CR3BP_MU, stm=False, batched=True)
    L2 = np.array([1 - CR3BP_MU + (CR3BP_MU / 3) ** (1/3), 0., 0.])
    colors = ["tab:blue", "tab:cyan", "tab:orange", "tab:red"]

    max_iter   = len(phase_history)
    snap_iters = [0, max_iter // 2, max_iter - 1]
    col_labels = ["Beginning", "Middle", "End"]

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    axes = [
        fig.add_subplot(gs[0, 1:3]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]

    for ax, snap_idx, col_label in zip(axes, snap_iters, col_labels):
        phases = phase_history[snap_idx]

        _, states = gen_state_history(ta=ta, initial_state=initial_state,
                                      time=period, n_points=n_points,
                                      phase=tuple(phases))

        for i, (orbit, color) in enumerate(zip(states, colors)):
            ax.plot(orbit[:, 1], orbit[:, 2], color=color, lw=0.5, alpha=0.3)
            ax.scatter(orbit[0, 1], orbit[0, 2], color=color, s=60,
                       label=f"obs {i+1}  ($\\phi$={phases[i]:.3f})")

        ax.scatter(L2[1], L2[2], marker="*", s=80, color="gold",
                   edgecolors="black", lw=0.5, label="L2")
        ax.set_title(f"{col_label}  (iter {snap_idx})")
        ax.set_xlabel("y (DU)")
        ax.set_ylabel("z (DU)")
        ax.legend(fontsize=7, loc="center", bbox_to_anchor=(0, 0.2), bbox_transform=ax.transData)
        ax.grid(True)

    plt.show()


def plot_configuration(phases):
    orbit_pos_z = [9.2828906799831279E-1, -6.2106134091203751E-26,  2.9579500586038443E-1,  3.8020223639592665E-11,  8.1841658932217537E-2, -1.7711888929326446E-10]
    orbit_neg_z = [9.2828906799831279E-1, -6.2106134091203751E-26, -2.9579500586038443E-1,  3.8020223639592665E-11,  8.1841658932217537E-2,  1.7711888929326446E-10]
    initial_state = np.array([orbit_pos_z, orbit_pos_z, orbit_neg_z, orbit_neg_z])

    period  = 2.1721303582818936
    n_points = 215

    ta = build_taylor_cr3bp(mu=CR3BP_MU, stm=False, batched=True)

    _, states = gen_state_history(ta=ta, initial_state=initial_state,
                                  time=period, n_points=n_points,
                                  phase=tuple(phases))   # (4, T, 6)

    L2 = np.array([1 - CR3BP_MU + (CR3BP_MU / 3) ** (1/3), 0., 0.])

    dot_colors   = ["#1565c0", "#00bcd4", "#b71c1c", "#ff8f00"]
    trace_colors = ["#90caf9", "#80deea", "#ef9a9a", "#ffcc80"]

    def _draw_arrow(ax, origin, direction, color, total_length=0.04, shaft_frac=0.6,
                    cone_radius=0.008, shaft_radius=0.001, n=20):
        d = direction / np.linalg.norm(direction)
        perp = np.array([1., 0., 0.]) if abs(d[0]) < 0.9 else np.array([0., 1., 0.])
        u = np.cross(d, perp); u /= np.linalg.norm(u)
        v = np.cross(d, u)
        theta = np.linspace(0, 2 * np.pi, n + 1)

        shaft_len  = total_length * shaft_frac
        cone_len   = total_length * (1 - shaft_frac)
        shaft_end  = origin + d * shaft_len

        # --- shaft (cylinder) ---
        t = np.linspace(0, 1, 2)
        T, Theta = np.meshgrid(t, theta)
        SX = origin[0] + d[0]*T*shaft_len + shaft_radius*(np.cos(Theta)*u[0] + np.sin(Theta)*v[0])
        SY = origin[1] + d[1]*T*shaft_len + shaft_radius*(np.cos(Theta)*u[1] + np.sin(Theta)*v[1])
        SZ = origin[2] + d[2]*T*shaft_len + shaft_radius*(np.cos(Theta)*u[2] + np.sin(Theta)*v[2])
        ax.plot_surface(SX, SY, SZ, color=color, linewidth=0, antialiased=True)

        # --- cone head ---
        T2, Theta2 = np.meshgrid(t, theta)
        r = cone_radius * (1 - T2)
        CX = shaft_end[0] + d[0]*T2*cone_len + r*(np.cos(Theta2)*u[0] + np.sin(Theta2)*v[0])
        CY = shaft_end[1] + d[1]*T2*cone_len + r*(np.cos(Theta2)*u[1] + np.sin(Theta2)*v[1])
        CZ = shaft_end[2] + d[2]*T2*cone_len + r*(np.cos(Theta2)*u[2] + np.sin(Theta2)*v[2])
        ax.plot_surface(CX, CY, CZ, color=color, linewidth=0, antialiased=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, (orbit, dot_color, trace_color) in enumerate(zip(states, dot_colors, trace_colors)):
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], color=trace_color, lw=1.5, alpha=0.8)
        ax.scatter(orbit[0, 0], orbit[0, 1], orbit[0, 2], color=dot_color, s=35,
                   label="observer" if i == 0 else "")

        if i in (0, 2):
            tangent = orbit[2, :3] - orbit[0, :3]
            _draw_arrow(ax, orbit[1, :3], tangent, color="black")

    ax.scatter(L2[0], L2[1], L2[2], marker="*", s=200, color="gold", edgecolors="black", lw=0.5, label="L2")

    ax.set_xlabel("x (DU)"); ax.set_ylabel("y (DU)"); ax.set_zlabel("z (DU)")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=8, label="observer"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=10, label="L2")]
    ax.legend(handles=handles, fontsize=9, loc="upper left", bbox_to_anchor=(0.48, 0.98))
    ax.set_aspect("equal")
    ax.view_init(elev=11, azim=-26)
    plt.show()


def _gd_loop_worker(_):
    import os, contextlib
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        phase, _ = gd_loop()
    return np.array(phase)


def run_histogram(n_runs=100):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    all_final_phases = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_gd_loop_worker, i) for i in range(n_runs)]
        for done, f in enumerate(as_completed(futures), 1):
            all_final_phases.append(f.result())
            print(f"\rCompleted {done}/{n_runs}", end="", flush=True)
    print()

    all_final_phases = np.array(all_final_phases)  # (n_runs, 4)
    optimized = all_final_phases[:, 1:]             # exclude fixed phase[0]=0.5

    from itertools import permutations
    targets = [0.0, 0.25, 0.75]
    perms = list(permutations(targets))             # 6 permutations

    def circ_dist(a, b):
        d = abs(a - b) % 1
        return min(d, 1 - d)

    def classify(row):
        rounded = tuple(targets[np.argmin([circ_dist(v, t) for t in targets])] for v in row)
        return perms.index(rounded) if rounded in perms else -1

    counts = np.zeros(len(perms), dtype=int)
    for row in optimized:
        idx = classify(row)
        if idx >= 0:
            counts[idx] += 1

    perm_labels = [str((0.5,) + p) for p in perms]

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(range(len(perms)), counts, color="steelblue", edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(perms)))
    ax.set_xticklabels(perm_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_xlabel("permutation")
    ax.set_title(f"Convergence to Each Permutation of (0, 0.25, 0.75)  —  {n_runs} runs")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    plt.show()
    return all_final_phases


if __name__ == "__main__":

    # phases, phase_history = gd_loop()
    # plot_phase_evolution(phase_history)
    # plot_configuration(np.random.uniform(0, 1, 4))
    run_histogram(100)