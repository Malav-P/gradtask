import jax
import jax.numpy as jnp

S = jnp.eye(2, 3)
e3 = jnp.array([0, 0, 1])
e1 = jnp.array([1, 0, 0])
e2 = jnp.array([0, 1, 0])


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


def _get_obs_jacobian(x, y):
    """
    Compute the observation Jacobian matrix H for a given state x and observer y.

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
