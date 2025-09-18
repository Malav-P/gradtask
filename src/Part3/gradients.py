import numpy as np
from .dynamics import cr3bp_vec
from .constants import CR3BP_MU

import jax
import jax.numpy as jnp
from functools import partial

def select_gradients(gradients, x):
    """
    Select the relevant gradients as directed by the solution to the assignment problem, x

    Args:
        gradients (np.ndarray[float]): array of shape (T, N, M, state_dim) of the gradients w.r.t position of satellites
        x (np.ndarray[int]): array of shape (T, N, M) of the assignments of observers to targets

    Returns:
        active gradients (np.ndarray): array of shape (N, T, state_dim)
    """

    # T, N, M, state_dim = gradients.shape

    # Mask the gradients by the assignment matrix
    # This keeps only the gradients of assigned targets
    masked_gradients = gradients * x[..., np.newaxis]  # shape: (T, N, M, state_dim)

    # Sum over the M targets to get total gradient per satellite per timestep
    summed_gradients = np.sum(masked_gradients, axis=2)  # shape: (T, N, state_dim)

    # Transpose to shape (N, T, state_dim)
    active_gradients = summed_gradients.transpose(1, 0, 2)

    return active_gradients


def compute_information(states_x, states_y, H_func, R_func, type='logdet', compute_grad=True):
    """
    Compute the information metric and its gradient w.r.t states_x using JAX.

    Args:
        states_x (np.ndarray): array of shape (N, T, state_dim)
        states_y (np.ndarray): array of shape (M, T, state_dim)
        H_func (callable): function(states_x, states_y) -> observation jacobian H
        R_func (callable): function(states_x, states_y) -> observation noise covariance R
        type (str): 'logdet' or 'trace'
        compute_grad (bool): whether to return gradient

    Returns:
        info: array of shape (T, N, M)
        grad: array of shape (T, N, M, state_dim) or None
    """

    states_x = jnp.array(states_x)
    states_y = jnp.array(states_y)

    # Single-pair info function
    def info_matrix(x, y):
        H = H_func(x, y)
        R = R_func(x, y)
        info_mat = H.T @ jnp.linalg.solve(R, H)
        if type == "det":
            _, det = jnp.linalg.slogdet(info_mat)
            return det
        elif type == "trace":
            return jnp.trace(info_mat)
        else:
            raise ValueError("type must be 'det' or 'trace'")

    # Gradient wrt x explicitly
    grad_fn = jax.grad(info_matrix, argnums=0) if compute_grad else None

    # Vectorize info
    v_info = jax.vmap(          # over T
        jax.vmap(              # over N
            jax.vmap(info_matrix, in_axes=(None, 0)),  # over M
            in_axes=(0, None)
        ),
        in_axes=(1, 1)  # time dimension
    )

    # Vectorize grad
    if compute_grad:
        v_grad = jax.vmap(
            jax.vmap(
                jax.vmap(grad_fn, in_axes=(None, 0)),
                in_axes=(0, None)
            ),
            in_axes=(1, 1)
        )
    else:
        v_grad = None

    @jax.jit
    def run(states_x, states_y):
        info = v_info(states_x, states_y)
        grad = v_grad(states_x, states_y) if compute_grad else None

        return info, grad
    
    info, grad = run(states_x, states_y)
    
    info = np.array(info)
    grad = np.array(grad) if compute_grad else None

    return info, grad


def compute_generalized_distances(states_x,
                                  states_y,
                                  Q,
                                  compute_grad=True):
    
    """
    Compute pairwise generalized distances of x to y of the form (x-y)^T Q (x-y) where Q is a diagonal matrix. Optionally, return the gradient w.r.t. x

    Args:
        states_x (np.ndarray): array of shape (N, T, 6) representing states of x 
        states_y (np.ndarray): array of shape (M, T, 6) representing states of y
        Q (np.ndarray): array of shape (6,) representing the diagonal of the matrix Q
        compute_grad (bool): Whether or not to return the gradient, default True
    Returns:
        dist (np.ndarray): array of shape (T, N, M). dist[i, j, k] is the distance from object j to object k at time i
        grad (np.ndarray): array of shape (T, N, M, 6) of the gradients w.r.t x
    """

    xQx = (Q * (states_x**2)).sum(axis=-1).T # (T, N)
    yQy = (Q * (states_y**2)).sum(axis=-1).T # (T, M)
    xQy = np.einsum('k,ntk,mtk->tnm', Q, states_x, states_y) # (T, N, M)

    l = xQx[..., None] + yQy[:, None, :] - 2*xQy
    dist = l 

    if compute_grad:
        dldx = 2 * Q[None, None, None, :] * (states_x[:, None, ...] - states_y[None, :, ...]) # (N, M, T, 6)
        dldx = dldx.transpose((2, 0, 1, 3)) # (T, N, M, 6)
        grad = dldx
    else:
        grad = None 

    return dist, grad

def compute_projected_gradients(gradients, states, reduction='sum'):
    """
    Compute the projected gradient of each satellite along its orbit.

    Args:
        gradients (np.ndarray): array of shape (N, T, 6). N is the number of satellites, T is the number of timesteps,
                                and C=6 and is the state dimension. gradients[i, j, k] is the gradient of the weight w.r.t the k-th state element of the
                                i-th observer at the j-th timestep 
        states (np.ndarray): array of shape (N, T, 6) representing the state vectors of sattelites. states[i, j, k] is the k-th state element
                             of the i-th observer and the j-th timestep
        reduction (str): the reduction operation to apply along the time axis. Default sum

    Returns:
        projected (np.ndarray) : array of shape (N, T) or (N,) the projected gradient, depending on the reduction operation
    
    """

    sdot = cr3bp_vec(None, states, mu=CR3BP_MU)  # (N, T, 6)

    unit_direction_vectors = sdot / np.linalg.norm(sdot, axis=-1, keepdims=True)  # (N, T, 6)

    unreduced_projected_grad = np.einsum('ijk,ijk->ij', gradients, unit_direction_vectors) # (N, T)

    match reduction:
        case "sum":
            projected_grad = unreduced_projected_grad.sum(axis=-1)
        case "mean":
            projected_grad = unreduced_projected_grad.mean(axis=-1)
        case None:
            projected_grad = unreduced_projected_grad
        case _:
            raise ValueError("reduction must be one of 'sum' 'mean' or 'None'")

    return projected_grad

# def make_noisy(arr, mean=0, std=0.1):
#     """
#     Make the gradients noisy (to avoid local minima). noisy_g = g * (1 + noise)

#     Args:
#         arr (np.ndarray): array of gradients
#         mean (float): mean of noise
#         std (float): standard deviation of noise

#     Returns:
#         noised_arr: array of noisy gradients of same shape as arr
#     """

#     noise = np.random.normal(loc=mean, scale=std, size=arr.shape)

#     noised_arr = arr * (1 + noise)

#     return noised_arr



if __name__ == "__main__":
    N = 7
    T = 10
    reduction = 'sum'
    gradients = np.random.randn(N, T, 6)
    states = np.random.randn(N, T, 6)

    projected_grad = compute_projected_gradients(gradients, states, reduction)


    M = 11
    states_x = np.random.randn(N, T, 6)
    states_y = np.random.randn(M, T, 6)

    dist, grad = compute_generalized_distances(states_x, states_y)

    print(dist.shape)
    print(grad.shape)