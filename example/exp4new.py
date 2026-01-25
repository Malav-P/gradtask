# from root : python3 -m example.exp4

from src.Part3.assignment_problem import solve_assignment_problem_time_expanded
from src.Part3.dynamics import gen_state_history, build_taylor_cr3bp
from src.Part3.gradients import (
    select_gradients,
    compute_generalized_distances,
    compute_projected_gradients,
)
from src.Part3.optimizers import SGD
from src.Part3.constants import CR3BP_MU

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})


def classify_two_minima(final_objs):
    """
    Classify runs into two minima based on final objective values.
    Returns labels in {0, 1}.
    """
    threshold = 0.15
    final_objs = np.asarray(final_objs)

    labels = (final_objs > threshold).astype(int)
    return labels, threshold



if __name__ == "__main__":

    # -----------------------------
    # Problem setup
    # -----------------------------
    ta = build_taylor_cr3bp(mu=CR3BP_MU, stm=False)

    initial_state = np.array(
        [
            1.1540242813087864,
            0.0,
            -0.1384196144071876,
            4.06530060663289e-15,
            -0.21493019200956867,
            8.48098638414804e-15,
        ]
    )

    initial_state_y = np.array(
        [
            0.8027692908754149,
            0.0,
            0.0,
            -1.1309830924549648e-14,
            0.33765564334938736,
            0.0,
        ]
    )

    time_ = 3.225
    n_points = 215
    max_iter = 100

    start_phase = np.array([0.9, 0.1])

    momenta = (0., 0., 0., 0.)
    noise_stds = (0., 0.1, 0.5, 1.5)

    n_runs = 100  # <<< number of repeated descents per sigma

    # -----------------------------
    # Precompute reference trajectory
    # -----------------------------
    _, states_y = gen_state_history(
        ta=ta,
        initial_state=np.tile(initial_state_y, (2, 1)),
        time=time_,
        n_points=n_points,
        phase=(0, 0.5),
    )

    # -----------------------------
    # Storage (per sigma)
    # -----------------------------
    all_obj = {}
    all_grad_norm = {}
    all_phase_diff = {}
    all_grad_sum = {}

    # -----------------------------
    # Main loop over sigmas
    # -----------------------------
    for momentum, noise in zip(momenta, noise_stds):

        obj_runs = []
        grad_norm_runs = []
        phase_diff_runs = []
        grad_sum_runs = []

        for run in range(n_runs):

            grad_history = np.empty((max_iter, 2))
            obj_history = np.empty(max_iter)
            phase_history = np.empty((max_iter, 2))

            optimizer = SGD(
                start_phase.copy(),
                modulo=1,
                momentum=momentum,
                lr=0.5,
                noise=noise,
            )

            for n_iter in range(max_iter):

                _, states_x = gen_state_history(
                    ta=ta,
                    initial_state=np.tile(initial_state, (2, 1)),
                    time=time_,
                    n_points=n_points,
                    phase=optimizer.parameters,
                )

                Q = np.array([1, 1, 1, 0, 0, 0])

                dist, grad = compute_generalized_distances(
                    states_x=states_x,
                    states_y=states_y,
                    Q=Q,
                    compute_grad=True,
                )

                x, obj = solve_assignment_problem_time_expanded(
                    weights=dist, opt_type="min"
                )

                obj_history[n_iter] = obj / n_points

                masked_g = select_gradients(grad, x)
                proj_g = compute_projected_gradients(
                    gradients=masked_g, states=states_x, reduction="mean"
                )

                grad_history[n_iter] = proj_g
                optimizer.step(proj_g)
                phase_history[n_iter] = optimizer.parameters


            # -------- collect derived metrics --------
            obj_runs.append(obj_history)
            grad_norm_runs.append(np.linalg.norm(grad_history, axis=-1))
            phase_diff_runs.append(
                np.abs(phase_history[:, 0] - phase_history[:, 1])
            )
            grad_sum_runs.append(
                np.abs(grad_history[:, 0] + grad_history[:, 1])
            )

            if noise == 0.0:
                break

        # stack: (n_runs, max_iter)
        all_obj[noise] = np.stack(obj_runs)
        all_grad_norm[noise] = np.stack(grad_norm_runs)
        all_phase_diff[noise] = np.stack(phase_diff_runs)
        all_grad_sum[noise] = np.stack(grad_sum_runs)

    # -----------------------------
    # Plotting helper
    # -----------------------------
    def plot_mean_and_band(fig_id, data_dict, ylabel):
        plt.figure(fig_id)

        for noise, data in data_dict.items():
            mean = data.mean(axis=0)
            std = data.std(axis=0)

            x = np.arange(mean.shape[0])

            plt.plot(x, mean, label=fr"$\sigma={noise}$")
            plt.fill_between(
                x,
                mean - 3 * std,
                mean + 3 * std,
                alpha=0.2,
            )

        plt.xlabel("Iteration number")
        plt.ylabel(ylabel)
        plt.legend(framealpha=0.5)
        plt.grid(True)
        plt.tight_layout()

    # -----------------------------
    # Figures
    # -----------------------------
    plot_mean_and_band(0, all_obj, "Time Averaged LP Objective")
    plot_mean_and_band(1, all_grad_norm, "Gradient norm")
    plot_mean_and_band(2, all_phase_diff, "Phase difference")
    plot_mean_and_band(3, all_grad_sum, "Grad component sum")

    

    # -----------------------------
    # Histogram of minima per noise
    # -----------------------------

    # -----------------------------
    # Single bar plot for all noise/momentum combinations
    # -----------------------------
    n_minima = 2
    n_sigmas = len(noise_stds[1:])

    # Prepare data
    counts_per_sigma = []

    for noise in noise_stds[1:]:
        final_objs = all_obj[noise][:, -1]
        labels, _ = classify_two_minima(final_objs)
        counts = np.bincount(labels, minlength=n_minima)
        counts_per_sigma.append(counts)

    counts_per_sigma = np.array(counts_per_sigma)  # shape: (n_sigmas, 2)

    # Plot
    x = np.arange(n_sigmas)  # the label locations for each noise/momentum
    width = 0.35  # width of each bar

    fig, ax = plt.subplots()

    bars1 = ax.bar(x - width/2, counts_per_sigma[:, 0], width, label="Global Min")
    bars2 = ax.bar(x + width/2, counts_per_sigma[:, 1], width, label="Local Min")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\sigma={s}$" for m, s in zip(momenta[1:], noise_stds[1:])])
    ax.set_ylabel("Count")
    ax.set_xlabel("Noise")
    ax.set_title("Converged Minimum Statistics")
    ax.legend()
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()


    plt.show()