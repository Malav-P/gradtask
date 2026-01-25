# from root : python3 -m example.exp7

# note: using computed information (using jax)
# note: need to clip gradients to avoid large changes in x after one step


from src.Part3.shortest_path_problem import solve_shortest_path_time_expanded, solve_shortest_path
from src.Part3.dynamics import gen_state_history, build_taylor_cr3bp
from src.Part3.gradients import select_gradients, compute_generalized_distances, compute_projected_gradients
from src.Part3.optimizers import SGD
from src.Part3.constants import Config, CR3BP_MU


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def gd_loop():
    initial_state = np.array([ 
                        0.8027692908754149,
                        0.0,
                        0.0,
                        -1.1309830924549648e-14,
                        0.33765564334938736,
                        0.0
                    ])

    config = Config(
        exp_name = "sp_exp1",
        period = 3.225,
        n_points = 215,
        max_iters = (50,),
        initial_state_observer= np.tile(initial_state, (2, 1)),
        initial_state_observer_phases=[(0., 0.5045)]                               
    )
    

    ta = build_taylor_cr3bp(mu=config.mu, stm=False, batched=True)
    n_satellites = len(config.initial_state_observer_phases[0])
    earth_x = np.array([-CR3BP_MU, 0, 0, 0, 0, 0])
    moon_x = np.array([1-CR3BP_MU, 0, 0, 0, 0, 0])

    earth_expanded = np.tile(earth_x, (config.n_points, 1))        # (T, 6)
    earth_expanded = earth_expanded[None, :, :]    # (1, T, 6)

    moon_expanded = np.tile(moon_x, (config.n_points, 1))
    moon_expanded = moon_expanded[None, :, :]

    x = np.empty(shape=(config.n_points, n_satellites+2, n_satellites+2))


    # Parameters
    A = 1.0        # amplitude
    mu = 0.0       # mean
    sigma = 1.0    # standard deviation

    # Generate x values
    xx = np.linspace(-5, 5, config.n_points)

    # Compute Gaussian values
    weights = A * np.exp(-((xx + 3 - mu)**2) / (2 * sigma**2))


    for p, max_iter in zip(config.initial_state_observer_phases, config.max_iters):

        grad_history = np.empty(shape=(max_iter, n_satellites))
        obj_history = np.empty(shape=(max_iter,))
        phase_history= np.empty(shape=(max_iter, n_satellites))

        optimizer = SGD((p[1],), modulo=1, momentum=0.0, lr=1, noise=0, clip_grad_norm=10, cosine_anneal=False, t_max=max_iter, eta_min=1e-4)

        for n_iter in range(max_iter):

            _, states_x = gen_state_history(ta=ta,
                                    initial_state=config.initial_state_observer,
                                    time=config.period,
                                    n_points=config.n_points,
                                    phase=(config.initial_state_observer_phases[0][0], optimizer.parameters[0]))
            



            states_x = np.concatenate([states_x, earth_expanded, moon_expanded], axis=0)
            dist, grad = compute_generalized_distances(states_x=states_x, states_y=states_x, Q=np.array([1, 1, 1, 0, 0, 0]))

            
            # objs = []
            # for t in range(config.n_points):
            #     x_var, path, OBJ = solve_shortest_path(dist[t], n_satellites, n_satellites + 1)
            #     objs.append(OBJ)
            #     x[t] = x_var
            # obj = np.average(objs)

            x, _, obj = solve_shortest_path_time_expanded(dist, n_satellites, n_satellites + 1)
            obj /= config.n_points

            masked_gradients = grad * x[..., np.newaxis]  # shape: (T, N, N, state_dim)

            g = np.empty(shape=(n_satellites, config.n_points, 6))
            for i in range(n_satellites):
                g[i] = np.sum(masked_gradients[:, i, :, :], axis=-2) - np.sum(masked_gradients[:, :, i, :], axis=-2)


            proj_g = compute_projected_gradients(g, states=states_x[:n_satellites], reduction="mean", weights=None)

            grad_history[n_iter] = proj_g

            optimizer.step(proj_g[1])

            phase = optimizer.parameters
            phase_history[n_iter] = phase

            print("Gradient: ", proj_g, "Objective: ", obj, "New Phases: ", phase)


def plot_example():
    initial_state = np.array([ 
                        0.8027692908754149,
                        0.0,
                        0.0,
                        -1.1309830924549648e-14,
                        0.33765564334938736,
                        0.0
                    ])

    config = Config(
        exp_name = "sp_exp1",
        period = 3.225,
        n_points = 215,
        max_iters = (1200,),
        initial_state_observer= np.tile(initial_state, (2, 1)),
        initial_state_observer_phases=[(0., 0.1)]                               
    )
    

    ta = build_taylor_cr3bp(mu=config.mu, stm=False, batched=True)
    n_satellites = len(config.initial_state_observer_phases[0])
    earth_x = np.array([-CR3BP_MU, 0, 0, 0, 0, 0])
    moon_x = np.array([1-CR3BP_MU, 0, 0, 0, 0, 0])

    earth_expanded = np.tile(earth_x, (config.n_points, 1))        # (T, 6)
    earth_expanded = earth_expanded[None, :, :]    # (1, T, 6)

    moon_expanded = np.tile(moon_x, (config.n_points, 1))
    moon_expanded = moon_expanded[None, :, :]

    x = np.empty(shape=(config.n_points, n_satellites+2, n_satellites+2))

    mean_cost = []
    mean_cost_gauss = []
    grads = []
    grads_gauss = []

    # Parameters
    A = 1.0        # amplitude
    mu = 0.0       # mean
    sigma = 1.0    # standard deviation

    # Generate x values
    xx = np.linspace(-5, 5, config.n_points)

    # Compute Gaussian values
    weights = A * np.exp(-((xx + 3 - mu)**2) / (2 * sigma**2))

    candidates = np.linspace(0.497, 0.5, 22)
    # candidates = [0.5, 0.498]
    # x_prev = None

    for p in candidates:

        _, states_x = gen_state_history(ta=ta,
                                initial_state=config.initial_state_observer,
                                time=config.period,
                                n_points=config.n_points,
                                phase=(config.initial_state_observer_phases[0][0], p))
        



        states_x = np.concatenate([states_x, earth_expanded, moon_expanded], axis=0)
        dist, grad = compute_generalized_distances(states_x=states_x, states_y=states_x, Q=np.array([1, 1, 1, 0, 0, 0]))


        
        objs = []
        for t in range(config.n_points):
            x_var, path, OBJ = solve_shortest_path(dist[t], n_satellites, n_satellites + 1)
            objs.append(OBJ)
            x[t] = x_var

        #     if x_prev is not None:
        #         s = (x_prev[t] == x[t]).sum() 
        #         if s != 16:
        #             print(s, t)

        # x_prev = x.copy()

        masked_gradients = grad * x[..., np.newaxis]  # shape: (T, N, N, state_dim)

        g = np.empty(shape=(n_satellites, config.n_points, 6))
        for i in range(n_satellites):
            g[i] = np.sum(masked_gradients[:, i, :, :], axis=-2) - np.sum(masked_gradients[:, :, i, :], axis=-2)


        proj_g = compute_projected_gradients(g, states=states_x[:n_satellites], reduction="mean", weights=None)
        proj_g_gauss = compute_projected_gradients(g, states=states_x[:n_satellites], reduction="mean", weights=weights)
        grads.append(proj_g[1])
        grads_gauss.append(proj_g_gauss[1])

        mean_cost.append(np.average(objs))
        mean_cost_gauss.append(np.average(objs, weights=weights))

        
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, constrained_layout=True)


    # Left: unweighted mean
    axes[0].plot(candidates, mean_cost, linewidth=1, alpha=0.2, color='black')
    axes[0].scatter(candidates, mean_cost, cmap='coolwarm', s=15, c=grads, vmin=-max(np.abs(grads)), vmax=max(np.abs(grads)))
    axes[0].set_xlabel(r"$\phi$")
    axes[0].set_ylabel(r"$\mathrm{Agg}\left(\text{LP}_{t}\right)$")
    axes[0].set_title("Mean Shortest Path Cost")
    axes[0].grid(True)

    cbar = plt.colorbar(
        mappable=axes[0].collections[0],
        ax=axes[0],
        label=r"$\mathrm{Agg}\left(\nabla_{\mathbf{s}}\text{LP}_{t}\right)$"
    )
    cbar.ax.yaxis.set_ticks_position('left')     # ticks on the left
    cbar.ax.yaxis.set_label_position('left') 


    # Right: weighted mean (Gaussian or whatever your weights are)
    axes[ 1].plot(candidates, mean_cost_gauss, linewidth=1, alpha=0.2, color='black')
    axes[ 1].scatter(candidates, mean_cost_gauss, cmap='coolwarm', s=15, c=grads_gauss, vmin=-max(np.abs(grads_gauss)), vmax = max(np.abs(grads_gauss)))
    axes[ 1].set_xlabel(r"$\phi$")
    axes[ 1].set_ylabel(r"$\mathrm{Agg}\left(\text{LP}_{t}\right)$")
    axes[ 1].set_title("Weighted Mean Shortest Path Cost")
    axes[ 1].grid(True)

    cbar = plt.colorbar(
        mappable=axes[1].collections[0],
        ax=axes[1],
        label=r"$\mathrm{Agg}\left(\nabla_{\mathbf{s}}\text{LP}_{t}\right)$"
    )

    cbar.ax.yaxis.set_ticks_position('left')     # ticks on the left
    cbar.ax.yaxis.set_label_position('left') 





    plt.show()



if __name__ == "__main__":

    gd_loop()
    # plot_example()