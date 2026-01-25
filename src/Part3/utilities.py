import jax
import jax.numpy as jnp
import numpy as np

S = jnp.eye(2, 3)
e3 = jnp.array([0, 0, 1])
e1 = jnp.array([1, 0, 0])
e2 = jnp.array([0, 1, 0])

indices = jnp.array([0, 1, 3, 4])  # x, y, xdot, ydot indices, use this if the orbit has not z component so that determinant is not 0



def _get_transformation_matrix(rho: jnp.ndarray, eps=1e-2):
    v3 = rho / jnp.linalg.norm(rho)

    v1_candidate = jnp.cross(e2, v3)
    norm_v1 = jnp.linalg.norm(v1_candidate)

    def use_v1_from_e2(_):
        return v1_candidate / norm_v1

    def use_v1_from_e1(_):
        v1_alt = jnp.cross(e1, v3)
        return v1_alt / jnp.linalg.norm(v1_alt)

    # Branch without Python "if"
    v1 = jax.lax.cond(norm_v1 > eps, use_v1_from_e2, use_v1_from_e1, operand=None)

    v2 = jnp.cross(v3, v1)
    v2 /= jnp.linalg.norm(v2)

    T_c = jnp.vstack((v1, v2, v3))
    return T_c

def angleanglerate_jacobian(x: jnp.ndarray, y: jnp.ndarray):
    """
    Compute the observation Jacobian matrix H = dh/dx for a given state x and observer y. 

    Args:
        x (jnp.ndarray):  observer state vector of shape (6,)
        y (jnp.ndarray):  target state vector of shape (6,)
    Returns:
        H (jnp.ndarray): Observation Jacobian matrix of shape (6, 6).
    """


    r_rel = y[0:3] - x[0:3]
    v_rel = y[3:6] - x[3:6]
    rnorm = jnp.linalg.norm(r_rel)
    H11 = jnp.eye(3)/rnorm - jnp.outer(r_rel,r_rel)/rnorm**3
    H21 = -jnp.outer(v_rel,r_rel) / rnorm**3 \
        -(jnp.outer(r_rel,v_rel) + jnp.dot(r_rel,v_rel)*jnp.eye(3)) / rnorm**3\
        + 3*jnp.dot(r_rel,v_rel) * jnp.outer(r_rel,r_rel) / rnorm**5
    return jnp.concatenate((
        jnp.concatenate((H11, jnp.zeros((3,3))), axis=1),
        jnp.concatenate((H21, H11), axis=1),
    ))


# def angleanglerate_R(x: jnp.ndarray, y: jnp.ndarray, sigma: float):
#     """
#     Compute the observation noise covariance R for angle-angle rate measurements.

#     Args:
#         x (jnp.ndarray): observer state vector of shape (6,)
#         y (jnp.ndarray): target state vector of shape (6,)
#         sigma (float): Standard deviation of the observation noise.

#     Returns:
#         R (jnp.ndarray): Observation noise covariance matrix of shape (6, 6).
#     """

#     r_rel = y[0:3] - x[0:3]
#     v_rel = y[3:6] - x[3:6]
    

#     return sigma**2 * jnp.block([
#         [jnp.dot(r_rel, r_rel) * jnp.eye(3) - jnp.outer(r_rel, r_rel), jnp.zeros((3,3))],
#         [jnp.zeros((3,3)), jnp.dot(v_rel, v_rel) * jnp.eye(3) - jnp.outer(v_rel, v_rel)]
#     ])

def _get_obs_jacobian(x, y):
    """
    Equation 7a of https://doi.org/10.1007/s40295-025-00520-8.
    Compute the observation Jacobian matrix H = dh/dx for a given state x and observer y. 

    Args:
        x (jnp.ndarray): State vector of shape (6,)
        y (jnp.ndarray): Observer state vector of shape (6,)
    Returns:
        H (jnp.ndarray): Observation Jacobian matrix of shape (4, 6).
    """
    rho = x[:3] - y[:3]
    nu = x[3:] - y[3:]
    T_c = _get_transformation_matrix(rho)

    rho_c = T_c @ rho
    nu_c = T_c @ nu

    # Stabilize division by z
    z = jnp.where(jnp.abs(rho_c[2]) < 1e-8, 1e-8, rho_c[2])

    A = (S / z) @ (jnp.eye(3) - jnp.outer(rho_c, e3) / z) @ T_c
    B = (-S / (z**2)) @ (
        nu_c[2] * jnp.eye(3)
        + jnp.outer(nu_c, e3)
        - 2 * (nu_c[2] / z) * jnp.outer(rho_c, e3)
    ) @ T_c

    H_top = jnp.concatenate([A, jnp.zeros((2,3))], axis=1)
    H_bottom = jnp.concatenate([B, A], axis=1)
    H = jnp.concatenate([H_top, H_bottom], axis=0)

    return H


def info_metric(x, y, H_func, R_func, type="det", stm_y=None):
    """
    Compute information metric for a single observer-target pair.

    Args:
        x: jnp array of shape (6,) state of observer
        y: jnp array of shape (6,) state of target
        H_func: Callable(jnp.array, jnp.array) -> observation jacobian H
        R_func: Callable(jnp.array, jnp.array) -> observation noise covariance R
        type: 'det' or 'trace'
        stm_y: jnp array of shape (6,6), state transition matrix of target (optional, not used here)

    Returns: 
        scalar information measure, either logdet or trace of the information matrix
    """
    H = H_func(x, y)
    R = R_func(x, y)
    info_mat = H.T @ jnp.linalg.solve(R, H)

    if stm_y is not None:
        info_mat = stm_y.T @ info_mat @ stm_y

    # extract on only x,y,xdot,ydot components
    M_sub = info_mat[jnp.ix_(indices, indices)]

    if type == "det":
        _, det = jnp.linalg.slogdet(M_sub)
        return det
    elif type == "trace":
        return jnp.trace(M_sub)
    else:
        raise ValueError("type must be 'det' or 'trace'")
    

def jit_vmap_info_metric(H_func, R_func, type="det"):
    """
    Returns a JIT-compiled, vectorized function that computes
    info_metric and its gradient w.r.t. observer state x.

    Args:
        H_func: Callable(jnp.array, jnp.array) -> observation jacobian H
        R_func: Callable(jnp.array, jnp.array) -> observation noise covariance R
        type: 'det' or 'trace'

    Returns:
        Callable that takes in (states_x, states_y, stm_y) and returns (info, grad) where:
            states_x (jnp.array): array of shape (N, T, state_dim)
            states_y (jnp.array): array of shape (M, T, state_dim)
            stm_y (jnp.array): Optional array of shape (M, T, state_dim, state_dim)
    """
    # JIT the scalar info_metric + grad
    info_and_grad = jax.jit(
        jax.value_and_grad(
            lambda x, y, stm_y: info_metric(x, y, H_func=H_func, R_func=R_func, type=type, stm_y=stm_y),
            argnums=0
        )
    )

    # Vectorize over [T, N, M] dimensions without swapping axes
    v_fn = jax.vmap(   # over T
        jax.vmap(       # over N
            jax.vmap(info_and_grad, in_axes=(None, 0, 0)),  # over M
            in_axes=(0, None, None)
        ),
        in_axes=(1, 1, 1)
    )

    return v_fn
