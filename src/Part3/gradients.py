import numpy as np

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

def compute_distances(states_x,
                      states_y,
                      compute_grad=True,
                      compute_squared=False,
                      only_use_positions=True,
                      eps=1e-4):
    """
    Compute pairwise distances of x to y. Optionally, return the gradient w.r.t. x

    Args:
        states_x (np.ndarray): array of shape (N, T, 6) representing states of x 
        states_y (np.ndarray): array of shape (M, T, 6) representing states of y
        compute_grad (bool): Whether or not to return the gradient, default True
        compute_squared (bool): Whether or not to return squared distances instead of just distance. Default False
        only_use_positions (bool): Whether or not to use the position vector only when computing distances (as opposed to full state vector). Default True
        eps (float) : divide by zero prevention. Default 1e-4

    Returns:
        dist (np.ndarray): array of shape (T, N, M). dist[i, j, k] is the distance from object j to object k at time i
        grad (np.ndarray): array of shape (T, N, M, 3) if only_use_positions is True otherwise (T, N, M, 6) of the gradients w.r.t x
    """
    x = states_x[..., :3] if only_use_positions else states_x
    y = states_y[..., :3] if only_use_positions else states_y

    x_squared = (x**2).sum(axis=-1).T # (T, N)
    y_squared = (y**2).sum(axis=-1).T # (T, M)
    xy    = np.einsum('ntk,mtk->tnm', x, y) # (T, N, M)

    l = x_squared[..., None] + y_squared[:, None, :] - 2*xy
    dist = np.sqrt(l) if compute_squared else l

    if compute_grad:
        dldx = 2 * (x[:, None, ...] - y[None, :, ...]) # (N, M, T, 6)
        dldx = dldx.transpose((2, 0, 1, 3)) # (T, N, M, 6)
        grad = dldx if compute_squared else (1 / (2 * np.sqrt(l+eps)))[..., None] * dldx 
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

    unit_direction_vectors = states[..., 3:] / np.linalg.norm(states[..., 3:], ord=2, axis=-1)[..., None] # (N, T, 3)

    unreduced_projected_grad = np.einsum('ijk,ijk->ij', gradients[..., :3], unit_direction_vectors) # (N, T)

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

    # print(projected_grad)


    M = 11
    states_x = np.random.randn(N, T, 6)
    states_y = np.random.randn(M, T, 6)

    dist, grad = compute_distances(states_x, states_y)

    print(dist.shape)
    print(grad.shape)