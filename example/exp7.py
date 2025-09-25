# from root : python3 -m example.exp2

# note: using computed information (using jax)


from src.Part3.assignment_problem import solve_assignment_problem
from src.Part3.dynamics import gen_state_history, build_taylor_cr3bp
from src.Part3.gradients import select_gradients, compute_information, compute_projected_gradients
from src.Part3.utilities import _get_obs_jacobian
from src.Part3.optimizers import SGD
from src.Part3.constants import CR3BP_MU

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
    start_phases = np.array([
                             
                             [0.9, 0.1]])   
    
    max_iters = (150,)
     
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

        optimizer = SGD(p, modulo=1, momentum=0, lr=0.0001, noise=0)

        for n_iter in range(max_iter):

            _, states_x = gen_state_history(ta=ta,
                                    initial_state=np.tile(initial_state, (2, 1)),
                                    time=time_,
                                    n_points=n_points,
                                    phase=optimizer.parameters)

            dist, grad = compute_information(states_x=states_x, states_y=states_y, H_func=_get_obs_jacobian, R_func= lambda x, y: jnp.eye(4), compute_grad=True, type='trace')  # (n_points, 2, 2) , (n_points, 2, 2, 3)

            grad *= -1  # we want to maximize information

            x = np.zeros_like(dist)
            obj = 0
            for i in range(n_points):
                x[i], objective = solve_assignment_problem(weights=dist[i], opt_type="max")
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