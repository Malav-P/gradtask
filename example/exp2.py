# from root : python3 -m example.exp2

# notes: we learn that ICs of the form (x0* + eps , x0* - eps) or (x1* + eps, x1* - eps) converge to local minima. The gradient is rubber banded to the optimal phase difference, but does not make progress to the optimal phase values.

from src.Part3.assignment_problem import solve_assignment_problem
from src.Part3.dynamics import gen_state_history, build_taylor_cr3bp
from src.Part3.gradients import select_gradients, compute_distances, compute_projected_gradients
from src.Part3.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

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
    


    
    for p, max_iter in zip(start_phases, max_iters):

        grad_history = np.empty(shape=(max_iter, 2))
        obj_history = np.empty(shape=(max_iter,))
        phase_history= np.empty(shape=(max_iter, 2))

        optimizer = SGD(p, modulo=1, momentum=0, lr=0.1, noise=0)

        for n_iter in range(max_iter):

            _, states_x = gen_state_history(ta=ta,
                                    initial_state=np.tile(initial_state, (2, 1)),
                                    time=time_,
                                    n_points=n_points,
                                    phase=optimizer.parameters)
            

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

            optimizer.step(proj_g)

            phase = optimizer.parameters

            phase_history[n_iter] = phase

            print("Gradient: ", proj_g, "Objective: ", obj/n_points, "New Phases: ", phase)

        plt.figure(0)
        plt.plot(obj_history, label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(1)
        plt.plot(np.linalg.norm(grad_history, axis=-1), label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(2)
        plt.plot(np.abs(phase_history[:,0] - phase_history[:,1]), label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(3)
        plt.plot(np.abs(grad_history[:,0] + grad_history[:,1]), label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

    
    plt.figure(0)
    plt.xlabel("Iteration number")
    plt.ylabel("Objective")
    plt.legend()
    # plt.savefig("media/exp2/obj.png")

    plt.figure(1)
    plt.xlabel("Iteration number")
    plt.ylabel("Gradient norm")
    plt.legend()
    # plt.savefig("media/exp2/gradnorm.png")

    plt.figure(2)
    plt.xlabel("Iteration number")
    plt.ylabel("Phase difference")
    plt.legend()
    # plt.savefig("media/exp2/phasediff.png")

    plt.figure(3)
    plt.xlabel("Iteration number")
    plt.ylabel("Grad component sum")
    plt.legend()


    plt.show()