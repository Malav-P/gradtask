# from root : python3 -m example.exp2

# notes: we learn that ICs of the form (x0* + eps , x0* - eps) or (x1* + eps, x1* - eps) converge to local minima. The gradient is rubber banded to the optimal phase difference, but does not make progress to the optimal phase values.
# global optimum is (0, 0.5) or (0.5, 0)

from blackboxphaseopt.assignment_problem import solve_assignment_problem_time_expanded
from blackboxphaseopt.dynamics import gen_state_history, build_taylor_cr3bp
from blackboxphaseopt.gradients import select_gradients, compute_generalized_distances, compute_projected_gradients
from blackboxphaseopt.optimizers import SGD
from blackboxphaseopt.constants import CR3BP_MU

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

def _safe_path(path):
    """Return path unchanged if it doesn't exist, otherwise insert _1, _2, … before the extension."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

def plot_orbit_snapshots(ta, initial_state, initial_state_y, time_, n_points, all_phase_histories, start_phases):
    """3-row × 3-col grid: rows = ICs, cols = begin / mid / end iteration."""
    n_ics = len(start_phases)
    fig, axes = plt.subplots(n_ics, 3, figsize=(13, 4 * n_ics))
    if n_ics == 1:
        axes = axes[np.newaxis, :]

    # Reference orbit traces (phase doesn't change the shape, only the starting point)
    _, orbit_x = gen_state_history(ta=ta, initial_state=np.tile(initial_state, (2, 1)),
                                   time=time_, n_points=n_points, phase=(0., 0.5))
    _, orbit_y = gen_state_history(ta=ta, initial_state=np.tile(initial_state_y, (2, 1)),
                                   time=time_, n_points=n_points, phase=(0., 0.5))

    obs_colors = ["tab:blue", "tab:orange"]
    col_labels = ["Beginning", "Middle", "End"]

    for row, (ph_hist, p0) in enumerate(zip(all_phase_histories, start_phases)):
        max_iter = len(ph_hist)
        snap_iters = [0, max_iter // 2, max_iter - 1]

        for col, snap_idx in enumerate(snap_iters):
            ax = axes[row, col]
            phases = ph_hist[snap_idx]

            _, states_snap = gen_state_history(ta=ta,
                                               initial_state=np.tile(initial_state, (2, 1)),
                                               time=time_, n_points=n_points,
                                               phase=phases)

            # Orbit traces
            ax.plot(orbit_x[0, :, 0], orbit_x[0, :, 1], color="silver", lw=1, zorder=1)
            ax.plot(orbit_y[0, :, 0], orbit_y[0, :, 1], color="silver", lw=1, ls="--", zorder=1)

            # Target positions
            for t_idx in range(orbit_y.shape[0]):
                ax.scatter(orbit_y[t_idx, 0, 0], orbit_y[t_idx, 0, 1],
                           marker="x", s=90, color="black", zorder=3,
                           label="target" if (row == 0 and col == 0 and t_idx == 0) else "")

            # Observer positions
            for sat_idx in range(states_snap.shape[0]):
                ax.scatter(states_snap[sat_idx, 0, 0], states_snap[sat_idx, 0, 1],
                           marker="o", s=80, color=obs_colors[sat_idx], zorder=3,
                           label=f"obs {sat_idx + 1} ($\\phi$={phases[sat_idx]:.2f})" if row == 0 and col == 0 else
                                 f"$\\phi$={phases[sat_idx]:.2f}")

            if row == 0:
                ax.set_title(f"{col_labels[col]}  (iter {snap_idx})")
            ax.set_xlabel("x (DU)")
            if col == 0:
                ax.set_ylabel(f"$\\phi_0$=[{p0[0]:.2f}, {p0[1]:.2f}]\ny (DU)")
            ax.set_aspect("equal")
            ax.legend(fontsize=12, loc="lower center")

    # plt.suptitle("Observer Phasing Evolution Along Orbit", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(_safe_path("media/exp2/phase_snapshots.png"), bbox_inches="tight")
    plt.show()


def plot_configuration(states_x, states_y, projection="xy"):
    plt.figure()

    match projection:
        case "xy":
            for i, orbit in enumerate(states_x):
                plt.plot(orbit[:, 0], orbit[:, 1], color="black")
                plt.scatter(orbit[[123], 0], orbit[[123],1], marker="s", color="black", label="" if i > 0 else "observer-init")
                if i == 0:
                    plt.scatter(orbit[[0, 107], 0], orbit[[0, 107],1], marker="s", color="red", label="" if i > 0 else "observer-optimized")

            for i, orbit in enumerate(states_y):
                plt.plot(orbit[:,0], orbit[:,1], color="black")
                plt.scatter(orbit[0, 0], orbit[0,1], marker="x", color="black", label= ""if i > 0 else "target")


            plt.xlabel("x (DU)")
            plt.ylabel("y (DU)")

            

        case "xz":
            for orbit in states_x:
                plt.plot(orbit[:, 0], orbit[:, 2])

            for orbit in states_y:
                plt.scatter(orbit[:,0], orbit[:,2])

            plt.xlabel("x (DU)")
            plt.ylabel("z (DU)")

        case "yz":
            for orbit in states_x:
                plt.plot(orbit[:, 1], orbit[:, 2])

            for orbit in states_y:
                plt.scatter(orbit[:,1], orbit[:,2])

            plt.xlabel("y (DU)")
            plt.ylabel("z (DU)")

    plt.legend(loc="lower center")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    ta = build_taylor_cr3bp(mu=CR3BP_MU, stm=False)
    initial_state = np.array([
                    1.1540242813087864,
                    0.0,
                    -0.1384196144071876,
                    4.06530060663289e-15,
                    -0.21493019200956867,
                    8.48098638414804e-15
                ])
    
    time_ = 3.225
    n_points = 215
    start_phases = np.array([[0., 0.1],
                             [0.9, 0.1],
                             [0.2, 0.6]])   
    
    max_iters = (50, 50, 50)

    # start_phases = np.array([
    #                          [0.2, 0.8]]) # 0.9, 0.1 -> local minima   0.901, 0.1 -> 0, 0.5  0.898, 0.1 -> 0.5, 0   if we make gradients noisy, this problem goes away
    
    # max_iters = (50,)
     
    initial_state_y = np.array([
                        0.8027692908754149,
                        0.0,
                        0.0,
                        -1.1309830924549648e-14,
                        0.33765564334938736,
                        0.0
                    ])
    
    _, states_y = gen_state_history(ta=ta,
                            initial_state=np.tile(initial_state_y, (2, 1)),
                            time=time_,
                            n_points=n_points,
                            phase=(0, 0.5))
    
    _, states_x = gen_state_history(ta=ta,
                        initial_state=np.tile(initial_state, (2, 1)),
                        time=time_,
                        n_points=n_points,
                        phase=(0, 0.5))

    data = {
        "targets": states_y[:, 0, :],
        "observers": states_x[:, 0, :]
    }

    
    all_phase_histories = []

    for p, max_iter in zip(start_phases, max_iters):

        grad_history = np.empty(shape=(max_iter, 2))
        obj_history = np.empty(shape=(max_iter,))
        phase_history= np.empty(shape=(max_iter, 2))

        optimizer = SGD(p, modulo=1, momentum=0, lr=0.5, noise=0)

        for n_iter in range(max_iter):

            _, states_x = gen_state_history(ta=ta,
                                    initial_state=np.tile(initial_state, (2, 1)),
                                    time=time_,
                                    n_points=n_points,
                                    phase=optimizer.parameters)
            

            # plot_configuration(states_x, states_y)
            # break

            Q = np.array([1, 1, 1, 0, 0, 0])

            dist, grad = compute_generalized_distances(states_x=states_x, states_y=states_y, Q=Q, compute_grad=True)  # (n_points, 2, 2) , (n_points, 2, 2, 3)

            # x = np.zeros_like(dist)
            # obj = 0
            # for i in range(n_points):
            #     x[i], objective = solve_assignment_problem(weights=dist[i], opt_type="min")
            #     obj += objective

            x, obj = solve_assignment_problem_time_expanded(weights=dist, opt_type="min")

            obj_history[n_iter] = obj / n_points

            masked_g = select_gradients(grad, x)   # (2, n_points, 3)

            proj_g = compute_projected_gradients(gradients=masked_g, states=states_x, reduction="mean")
            grad_history[n_iter] = proj_g

            optimizer.step(proj_g)

            phase = optimizer.parameters

            phase_history[n_iter] = phase

            print("Gradient: ", proj_g, "Objective: ", obj/n_points, "New Phases: ", phase, "Number of assignments: ", np.sum(x))

        all_phase_histories.append(phase_history.copy())

        plt.figure(0)
        plt.plot(obj_history, label=f'$\\phi_0$=[{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(1)
        plt.plot(np.linalg.norm(grad_history, axis=-1), label=f'$\\phi_0$=[{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(2)
        plt.plot(np.abs(phase_history[:,0] - phase_history[:,1]), label=f'$\\phi_0$=[{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(3)
        plt.plot(np.abs(grad_history[:,0] + grad_history[:,1]), label=f'$\\phi_0$=[{p[0]:.2f}, {p[1]:.2f}]')

    
    plt.figure(0)
    plt.xlabel("Iteration number")
    plt.ylabel("Time Averaged LP Objective")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(_safe_path("media/exp2/obj.png"))

    plt.figure(1)
    plt.xlabel("Iteration number")
    plt.ylabel("Gradient norm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_safe_path("media/exp2/gradnorm.png"))

    plt.figure(2)
    plt.xlabel("Iteration number")
    plt.ylabel("Phase difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_safe_path("media/exp2/phasediff.png"))

    plt.figure(3)
    plt.xlabel("Iteration number")
    plt.ylabel("Grad component sum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_orbit_snapshots(ta, initial_state, initial_state_y, time_, n_points,
                         all_phase_histories, start_phases)

    plt.show()