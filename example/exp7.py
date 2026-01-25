# from root : python3 -m example.exp7

# note: using computed information (using jax)
# note: need to clip gradients to avoid large changes in x after one step


from src.Part3.assignment_problem import solve_assignment_problem_time_expanded
from src.Part3.dynamics import gen_state_history, build_taylor_cr3bp
from src.Part3.gradients import select_gradients, compute_information, compute_projected_gradients
from src.Part3.utilities import _get_obs_jacobian, angleanglerate_jacobian, jit_vmap_info_metric
from src.Part3.optimizers import SGD
from src.Part3.constants import Config


import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


if __name__ == "__main__":

    initial_state_y = np.array([ 
                        0.8027692908754149,
                        0.0,
                        0.0,
                        -1.1309830924549648e-14,
                        0.33765564334938736,
                        0.0
                    ])
    
    initial_state = np.array([
                    1.1540242813087864,
                    0.0,
                    -0.1384196144071876,
                    4.06530060663289e-15,
                    -0.21493019200956867,
                    8.48098638414804e-15
                ])

    config = Config(
        exp_name = "exp7",
        period = 3.225,
        n_points = 215,
        max_iters = (1200,),
        initial_state_target = np.tile(initial_state_y, (2, 1)),
        initial_state_target_phases = (0, 0.5),
        initial_state_observer= np.tile(initial_state, (2, 1)),
        initial_state_observer_phases=[(0.75, 0.7)],
        t_expose=0.75 * 3.225 / 215,
        sigma = 3 * np.pi / 180, # observation uncertainty in radians for angle-anglerate measurement
        assignment_penalty_lambda=0
                               
    )
    


    ###### OPTICAL MEASUREMENT MODEL ###########

    # R = config.sigma**2 * jnp.block([[jnp.eye(2), jnp.zeros(shape=(2,2))], [jnp.zeros(shape=(2,2)), (2/(config.t_expose**2))*jnp.eye(2)]])
    # H_func = _get_obs_jacobian

    ########################################################

    ###### ANGLE-ANGLE RATE MEASUREMENT MODEL ###########

    R_func = lambda x, y : config.sigma**2 * jnp.concatenate((
    jnp.concatenate((jnp.eye(3), jnp.zeros((3,3))), axis=1),
    jnp.concatenate((jnp.zeros((3,3)), 2/config.t_expose**2 * jnp.eye(3)), axis=1),
    ))

    H_func = angleanglerate_jacobian

    ########################################################

    info_metric = jit_vmap_info_metric(H_func=H_func, R_func=R_func, type='det')
  
    
    ta = build_taylor_cr3bp(mu=config.mu, stm=False, batched=True)
    _, states_y = gen_state_history(ta=ta,
                            initial_state=config.initial_state_target,
                            time=config.period,
                            n_points=config.n_points,
                            phase=config.initial_state_target_phases)

    
    for p, max_iter in zip(config.initial_state_observer_phases, config.max_iters):

        grad_history = np.empty(shape=(max_iter, 2))
        obj_history = np.empty(shape=(max_iter,))
        phase_history= np.empty(shape=(max_iter, 2))

        optimizer = SGD(p, modulo=1, momentum=0, lr=0.001, noise=0, cosine_anneal=True, t_max=max_iter, eta_min=1e-4, clip_grad_norm=10)

        best_obj = -np.inf
        best_controls = None

        for n_iter in range(max_iter):

            _, states_x = gen_state_history(ta=ta,
                                    initial_state=config.initial_state_observer,
                                    time=config.period,
                                    n_points=config.n_points,
                                    phase=optimizer.parameters)

            dist, grad = compute_information(states_x=states_x, states_y=states_y, info_metric=info_metric)  # (n_points, 2, 2) , (n_points, 2, 2, 3)

            grad *= -1  # we want to maximize information

            x, obj = solve_assignment_problem_time_expanded(weights=dist, assignment_penalty_lambda=config.assignment_penalty_lambda, opt_type="max")


            obj_history[n_iter] = obj / config.n_points

            masked_g = select_gradients(grad, x)  
            proj_g = compute_projected_gradients(gradients=masked_g, states=states_x, reduction="mean")
            grad_history[n_iter] = proj_g

            optimizer.step(proj_g)

            phase = optimizer.parameters

            phase_history[n_iter] = phase

            if obj > best_obj:
                best_obj = obj
                best_controls = x

            print("Gradient: ", proj_g, "Objective: ", obj/config.n_points, "New Phases: ", phase, "Number of assignments", x.sum())


        # np.savez("exp7.npz", u=best_controls, optimal_observer_phases = optimizer.parameters, **asdict(config))


        plt.figure(0)
        plt.plot(obj_history, label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(1)
        plt.plot(np.linalg.norm(grad_history, axis=-1), label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(2)
        plt.plot(np.abs(phase_history[:,0] - phase_history[:,1]), label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

        plt.figure(3)
        plt.plot(np.abs(grad_history[:,0] + grad_history[:,1]), label=f'IC: [{p[0]:.2f}, {p[1]:.2f}]')

    
    plt.figure(0)
    plt.title("Cosine Annealing + Gradient Clipping")
    plt.xlabel("Iteration number")
    plt.ylabel("Time Averaged LP Objective")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # plt.savefig("media/exp2/obj.png")

    plt.figure(1)
    plt.xlabel("Iteration number")
    plt.ylabel("Gradient norm")
    plt.legend()
    # plt.savefig("media/exp2/gradnorm.png")

    plt.figure(2)
    plt.xlabel("Iteration number")
    plt.ylabel("Phase difference")
    plt.grid(True)
    plt.legend()
    # plt.savefig("media/exp2/phasediff.png")

    plt.figure(3)
    plt.xlabel("Iteration number")
    plt.ylabel("Grad component sum")
    plt.legend()


    plt.show()