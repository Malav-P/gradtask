# Simplified version: just compute states_x, states_y, and compute_information

from blackboxphaseopt.dynamics import gen_state_history, build_taylor_cr3bp
from blackboxphaseopt.gradients import compute_information, compute_projected_gradients
from blackboxphaseopt.utilities import angleanglerate_jacobian, jit_vmap_info_metric
from blackboxphaseopt.constants import Config

import numpy as np
import jax.numpy as jnp


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
        exp_name = "exp7_simple",
        period = 3.225,
        n_points = 215,
        max_iters = (1200,),
        initial_state_target = np.reshape(initial_state_y, (1,6)),
        initial_state_target_phases = (0,),
        initial_state_observer= np.reshape(initial_state, (1,6)),
        initial_state_observer_phases=[(0.2,)],
        t_expose=0.75 * 3.225 / 215,
        sigma = 3 * np.pi / 180,
        assignment_penalty_lambda=0
    )
    
    # Setup measurement model
    R_func = lambda x, y : config.sigma**2 * jnp.concatenate((
        jnp.concatenate((jnp.eye(3), jnp.zeros((3,3))), axis=1),
        jnp.concatenate((jnp.zeros((3,3)), 2/config.t_expose**2 * jnp.eye(3)), axis=1),
    ))

    H_func = angleanglerate_jacobian
    
    indices = jnp.array([0,1,3,4])  # select x, y, xdot, ydot
    info_metric = jit_vmap_info_metric(H_func=H_func, R_func=R_func, type='det', indices=indices)
    
    # Build dynamics
    ta = build_taylor_cr3bp(mu=config.mu, stm=False, batched=True)
    
    # Compute states_y (target)
    _, states_y = gen_state_history(ta=ta,
                            initial_state=config.initial_state_target,
                            time=config.period,
                            n_points=config.n_points,
                            phase=config.initial_state_target_phases)
    

    _, states_x = gen_state_history(ta=ta,
                            initial_state=config.initial_state_observer,
                            time=config.period,
                            n_points=config.n_points,
                            phase=config.initial_state_observer_phases[0])
    
    # Compute information
    dist, grad = compute_information(states_x=states_x, states_y=states_y, info_metric=info_metric)

    g = compute_projected_gradients(grad.sum(axis=2).transpose(1, 0, 2), states_x)
    
    print("states_x shape:", states_x.shape)
    print("states_y shape:", states_y.shape)
    print("dist shape:", dist.shape)
    print("grad shape:", grad.shape)
    # print("\nInformation (dist):\n", dist)
    # print("\nGradient:\n", grad)
    print(g)


    import matplotlib.pyplot as plt
    plt.plot(dist.flatten())
    plt.title("Information Metric over Time")
    

    # plot grad norm
    plt.figure()
    
    grad = grad.squeeze()
    threshold = 1000
    grad = threshold * np.tanh(grad / threshold)
    plt.plot(np.linalg.norm(grad, axis=-1))
    plt.show()