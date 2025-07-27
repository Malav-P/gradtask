# from root : python3 -m example.exp5


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
                plt.plot(orbit[:, 0], orbit[:, 1], color="black", linewidth=0.5)
                
                plt.scatter(orbit[[0], 0], orbit[[0],1], marker="s", color="black", label="" if i > 0 else "observer")

            for i, orbit in enumerate(states_y):
                plt.plot(orbit[:,0], orbit[:,1], color="black", linewidth=0.5)
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

    plt.legend()
    plt.tight_layout()
    plt.show()

seed=0 # seed = 0 -> smooth convergence to local minimum
np.random.seed(seed)


if __name__ == "__main__":

    mu = 1.215058560962404e-02
    time_ = 6.45
    n_points = 215
    tol=1e-9

    ta = build_taylor_cr3bp(mu=mu, stm=False)

    initial_state = np.array([[0.13603399956670137, 0, 0, 1.9130717669166003e-12, 3.202418276067991, 0], # 3:1
                              [0.9519486347314083, 0, 0, 0, -0.952445273435512, 0], # 2:1
                              [0.65457084231188, 0, 0, 3.887957091335523e-13, 0.7413347560791179, 0], # L1 Lyap
                              [0.9982702689023665, 0, 0, -2.5322340091977996e-14, 1.5325475708886613, 0] # L2 Lyap
                              ]) # TODO
    
    n_observers, _ = initial_state.shape
    
    N = 1 # number of phase vectors to try
    start_phases = np.random.random(size=(N, n_observers))  
    # start_phases = np.array([[0.8, 0.82, 0.8, 0.54]]) -> bouncy convergence
    # start_phases = np.array([[0.35, 0.63, 0.4, 0.52]]) #-> best convergence  
    

    info = np.zeros(shape=(N, 2*n_observers + 1))

    max_iters = (500,) * N
     
    initial_state_y = np.array([[0.8027692908754149, 0, 0, -1.1309830924549648e-14, 0.33765564334938736, 0], # L1 Lyap short
                                [1.1540242813087864, 0, -0.1384196144071876, 4.06530060663289e-15, -0.21493019200956867, 8.48098638414804e-15] # L2 Halo short
                                ]) 
    
    _, states_y = gen_state_history(ta=ta,
                            initial_state=initial_state_y,
                            time=time_,
                            n_points=n_points)
    
    for j, (p, max_iter) in enumerate(zip(start_phases, max_iters)):

        grad_history = np.empty(shape=(max_iter, n_observers))
        obj_history = np.empty(shape=(max_iter,))
        phase_history= np.empty(shape=(max_iter, n_observers))

        optimizer = SGD(p, modulo=1, momentum=0, lr=0.1, noise=0.1)

        num_steps=0
        while num_steps < max_iter:

            _, states_x = gen_state_history(ta=ta,
                                    initial_state=initial_state,
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

            obj_history[num_steps] = obj / n_points

            masked_g = select_gradients(grad, x)  
            proj_g = compute_projected_gradients(gradients=masked_g, states=states_x, reduction="mean")
            grad_history[num_steps] = proj_g

            optimizer.step(proj_g)

            phase = optimizer.parameters

            phase_history[num_steps] = phase

            print(f"Gradient: {proj_g}, Objective: {obj/n_points:.3f}, New Phases: {phase}")

            num_steps+=1

            if np.all(np.abs(proj_g) < tol):
                print(f"\n\n\nITERATION {j} COMPLETED IN {num_steps}\n\n\n")
                break

        
        info[j, :n_observers] = p
        info[j, n_observers:-1] = optimizer.parameters
        info[j, -1] = obj/n_points

        plt.figure(0)
        plt.plot(obj_history[:num_steps], label=f"IC: {[f'{i:.2f}' for i in p]}")

        plt.figure(1)
        plt.plot(np.linalg.norm(grad_history[:num_steps], axis=-1), label=f"IC: {[f'{i:.2f}' for i in p]}")

    
    # np.savetxt("converged_info.txt", info, fmt="%.6f")  # Save with 6 decimal places

    plt.figure(0)
    plt.xlabel("Iteration number")
    plt.ylabel("Objective")
    plt.legend()
    # # plt.savefig("media/exp5/obj.png")

    plt.figure(1)
    plt.xlabel("Iteration number")
    plt.ylabel("Gradient norm")
    plt.legend()
    # # plt.savefig("media/exp5/gradnorm.png")


    plt.show()