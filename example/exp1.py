# from root : python3 -m example.exp1

# two observers on an L2 halo orbit and two static target points at (1, +- 0.2, 0)

from src.Part3.assignment_problem import solve_assignment_problem
from src.Part3.dynamics import gen_state_history, build_taylor_cr3bp
from src.Part3.gradients import select_gradients, compute_distances, compute_projected_gradients

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})


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
                plt.scatter(orbit[:,0], orbit[:,1], marker="x", color="black", label= ""if i > 0 else "target")


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
    alpha = 0.1
    mu = 1.215058560962404e-02
    
    ta = build_taylor_cr3bp(mu=mu, stm=False)
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
    
    max_iter = 20
    grad_history = np.zeros(shape=(max_iter, 2))
    obj_history = np.zeros(shape=(max_iter,))
    phase_history= np.zeros(shape=(max_iter, 2))

    base_states = np.array([[1.0, 0.2, 0., 0., 0., 0.],
                            [1.0,-0.2, 0., 0., 0., 0.]])

    states_y = np.tile(base_states[:, np.newaxis, :], (1, n_points, 1))


    
    
    for p in start_phases:

        phase = np.copy(p)

        for n_iter in range(max_iter):

            _, states_x = gen_state_history(ta=ta,
                                    initial_state=np.tile(initial_state, (2, 1)),
                                    time=time_,
                                    n_points=n_points,
                                    phase=phase)
            

            # plot_configuration(states_x, states_y)

            # break


            dist, grad = compute_distances(states_x=states_x, states_y=states_y, compute_grad=True, compute_squared=False)  # (n_points, 2, 2) , (n_points, 2, 2, 3)

            x = np.zeros_like(dist)
            obj = 0
            for i in range(n_points):
                x[i], objective = solve_assignment_problem(weights=dist[i], opt_type="min")
                obj += objective

            obj_history[n_iter] = obj / n_points

            masked_g = select_gradients(grad, x)   # (2, n_points, 3)
            proj_g = compute_projected_gradients(gradients=masked_g, states=states_x, reduction="mean")
            grad_history[n_iter] = proj_g

            phase -= alpha * proj_g  # gradient descent step

            phase = phase % 1  # keep in range (0, 1]

            phase_history[n_iter] = phase

            print("Gradient: ", proj_g, "Objective: ", obj/n_points, "New Phases: ", phase)

        plt.figure(0)
        plt.plot(obj_history, label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(1)
        plt.plot(np.linalg.norm(grad_history, axis=-1), label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(2)
        plt.plot(np.abs(phase_history[:,0] - phase_history[:,1]), label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')
    
    plt.figure(0)
    plt.xlabel("Iteration number")
    plt.ylabel("Objective")
    plt.legend()
    plt.tight_layout()

    # plt.savefig("media/exp1/objexp1.png")

    plt.figure(1)
    plt.xlabel("Iteration number")
    plt.ylabel("Gradient norm")
    plt.legend()
    plt.tight_layout()

    # plt.savefig("media/exp1/gradnormexp1.png")

    plt.figure(2)
    plt.xlabel("Iteration number")
    plt.ylabel("Phase difference")
    plt.legend()
    plt.tight_layout()

    # plt.savefig("media/exp1/phasediffexp1.png")


    plt.show()