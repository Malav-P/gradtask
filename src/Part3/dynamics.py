import numpy as np
import heyoka as hy

def cr3bp(t, s, mu):
    """
    Computes the derivatives of the state vector for the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
        t (float): Time parameter (not used in this function).
        s (np.ndarray[float]): State vector [x, y, z, vx, vy, vz].
        mu (float): Mass ratio of the CR3BP system.

    Returns:
        np.ndarray[float]: Derivatives of the state vector.

    Notes:
        - The CR3BP models motion in a rotating coordinate system where two primary bodies (e.g., Earth and Moon) 
          move in circular orbits around their center of mass while a third, negligible mass moves under 
          their gravitational influence.
        - The function computes the derivatives of the state vector with respect to time.

    """
    x, y, z, vx, vy, vz = s

    ds = np.array([vx, vy, vz, 0, 0, 0])

    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)

    ds[3] = 2*vy + x - ((1-mu)/r1**3)*(x+mu) + (mu/r2**3)*(1-mu-x)
    ds[4] = -2*vx + y - ((1-mu)/r1**3)*y - (mu/r2**3)*y
    ds[5] = -((1-mu)/r1**3)*z - (mu/r2**3)*z

    return ds

def jac_cr3bp(t, s, mu):
    """
    Computes the Jacobian matrix of the Circular Restricted Three-Body Problem (CR3BP) dynamics.

    Parameters:
        t (float): Time parameter (not used in this function).
        s (np.ndarray[float]): State vector [x, y, z, vx, vy, vz].
        mu (float): Mass ratio of the CR3BP system.

    Returns:
        np.ndarray[float]: Jacobian matrix.

    Notes:
        - The CR3BP models motion in a rotating coordinate system where two primary bodies (e.g., Earth and Moon) 
          move in circular orbits around their center of mass while a third, negligible mass moves under 
          their gravitational influence.
        - The Jacobian matrix represents the linearization of the system dynamics around the given state.
        - The function computes the Jacobian matrix A, where A = [[0, I], [U, 2*omega]], with I being the identity matrix and
          U being the gravitational potential matrix.
    """
    x, y, z, vx, vy, vz = s

    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)

    omega = np.array([[0 , 1, 0],
                      [-1, 0, 0],
                      [0 , 0, 0]])
    
    I = np.eye(3)

    U_xx = 1 - (1-mu)*(r1**2 - 3*(x+mu)**2) / r1**5 - mu*(r2**2 - 3*(x - (1-mu))**2) / r2**5
    U_yy = 1 - (1-mu)*(r1**2 - 3*(y)**2) / r1**5 - mu*(r2**2 - 3*(y)**2) / r2**5
    U_zz = -(1-mu)*(r1**2 - 3*(z)**2) / r1**5 - mu*(r2**2 - 3*(z)**2) / r2**5 

    U_xy = 3*y*(1-mu)*(x+mu) / r1**5 + 3*mu*y*(x - (1-mu)) / r2**5
    U_xz = 3*z*(1-mu)*(x+mu) / r1**5 + 3*mu*z*(x - (1-mu)) / r2**5
    U_yz = 3*y*z*(1-mu) / r1**5 + 3*mu*y*z / r2**5

    U = np.array([[U_xx, U_xy, U_xz],
                  [U_xy, U_yy, U_yz],
                  [U_xz, U_yz, U_zz]])
    Z = np.zeros(shape=(3,3))

    A = np.block([[Z, I],
                  [U, 2*omega]])
    
    return A

def build_taylor_cr3bp(mu, stm=False, batched=True):
    """
    Build Taylor integrator for CR3BP equations of motion.
    If STM option is `True`, the state-vector is length-42 (6 states, 6x6 STM, row-by-row). 
    Args:
        mu (float): CR3BP gravitational parameter
        stm (bool): whether to include STM, default is False
        batched (bool): whether to return the batched integrator, default it True
        
    Returns:
        ta (hy.taylor_adaptive): taylor integrator for CR3BP
    """
    # parameters
    mu_param = hy.par[0]

    # Variables
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

    # compute r1, r2s
    r1_sq = (x+mu_param)**2 + y**2 + z**2
    r2_sq = (x-1+mu_param)**2 + y**2 + z**2
    r1 = hy.sqrt( r1_sq )
    r2 = hy.sqrt( r2_sq )

    # construct ODE system
    if stm is False:
        # temporary initial condition
        tmp_ic = [1.0, 0.0, 0.1, 0.0, 1.0, 0.0]
        ode_sys = [
            (x, vx),
            (y, vy),
            (z, vz),
            (vx,  2*vy + x - ((1-mu_param)/r1**3)*(mu_param+x) + (mu_param/r2**3)*(1-mu_param-x)),
            (vy, -2*vx + y - ((1-mu_param)/r1**3)*y      - (mu_param/r2**3)*y),
            (vz,            -((1-mu_param)/r1**3)*z      - (mu_param/r2**3)*z),
        ]

    else:
        # make additional variables
        s6,  s7,  s8,  s9,  s10, s11 = hy.make_vars("s6", "s7", "s8", "s9", "s10", "s11")
        s12, s13, s14, s15, s16, s17 = hy.make_vars("s12", "s13", "s14", "s15", "s16", "s17")
        s18, s19, s20, s21, s22, s23 = hy.make_vars("s18", "s19", "s20", "s21", "s22", "s23")
        s24, s25, s26, s27, s28, s29 = hy.make_vars("s24", "s25", "s26", "s27", "s28", "s29")
        s30, s31, s32, s33, s34, s35 = hy.make_vars("s30", "s31", "s32", "s33", "s34", "s35")
        s36, s37, s38, s39, s40, s41 = hy.make_vars("s36", "s37", "s38", "s39", "s40", "s41")

        # temporary initial condition
        tmp_ic = 7*[1.0, 0.0, 0.1, 0.0, 1.0, 0.0]
        # coefficients of A matrix
        # first ~ third rows
        a00, a01, a02, a03, a04, a05 = 0, 0, 0, 1, 0, 0
        a10, a11, a12, a13, a14, a15 = 0, 0, 0, 0, 1, 0
        a20, a21, a22, a23, a24, a25 = 0, 0, 0, 0, 0, 1
        # fourth ~ sixth rows
        a33, a34, a35 =  0, 2, 0
        a43, a44, a45 = -2, 0, 0
        a53, a54, a55 =  0, 0, 0

        # pre-compute coefficients that are re-used
        d1 = (1-mu_param)/( r1_sq )**1.5
        d2 = mu_param/(r2_sq)**1.5
        d3 = 3*((x+mu_param)**2 *(1-mu_param)/( r1_sq )**2.5 + (x+mu_param-1)**2*mu_param/(r2_sq)**2.5)
        d4 = ( 3*y*((x+mu_param)*(1-mu_param)/( r1_sq )**2.5 + (x+mu_param-1)*mu_param/(r2_sq)**2.5) )
        d5 = ( 3*z*((x+mu_param)*(1-mu_param)/( r1_sq )**2.5 + (x+mu_param-1)*mu_param/(r2_sq)**2.5) )
        d6 = 3*y**2*((1-mu_param)/( r1_sq )**2.5 + mu_param/(r2_sq)**2.5)
        d7 = ( 3*y*z*((1-mu_param)/( r1_sq )**2.5 + mu_param/(r2_sq)**2.5) )
        d8 = 3*z**2*((1-mu_param)/( r1_sq )**2.5 + mu_param/(r2_sq)**2.5)

        dsum_1 = ( 1 - d1 - d2 + d3 )
        dsum_2 = ( 1 - d1 - d2 + d6 )
        dsum_3 = ( -d1 - d2 + d8 )

        # STATE
        ode_sys = [
            (x, vx),
            (y, vy),
            (z, vz),
            (vx,  2*vy + x - ((1-mu_param)/r1**3)*(mu_param+x) + (mu_param/r2**3)*(1-mu_param-x)),
            (vy, -2*vx + y - ((1-mu_param)/r1**3)*y      - (mu_param/r2**3)*y),
            (vz,            -((1-mu_param)/r1**3)*z      - (mu_param/r2**3)*z),
            # STATE-TRANSITION MATRIX
            # first row ... 
            (s6,  a00*s6  + a01*s12 + a02*s18 + a03*s24 + a04*s30 + a05*s36),
            (s7,  a00*s7  + a01*s13 + a02*s19 + a03*s25 + a04*s31 + a05*s37),
            (s8,  a00*s8  + a01*s14 + a02*s20 + a03*s26 + a04*s32 + a05*s38), 
            (s9,  a00*s9  + a01*s15 + a02*s21 + a03*s27 + a04*s33 + a05*s39),
            (s10, a00*s10 + a01*s16 + a02*s22 + a03*s28 + a04*s34 + a05*s40),
            (s11, a00*s11 + a01*s17 + a02*s23 + a03*s29 + a04*s35 + a05*s41),
            # second row ...
            (s12, a10*s6  + a11*s12 + a12*s18 + a13*s24 + a14*s30 + a15*s36),
            (s13, a10*s7  + a11*s13 + a12*s19 + a13*s25 + a14*s31 + a15*s37),
            (s14, a10*s8  + a11*s14 + a12*s20 + a13*s26 + a14*s32 + a15*s38),
            (s15, a10*s9  + a11*s15 + a12*s21 + a13*s27 + a14*s33 + a15*s39),
            (s16, a10*s10 + a11*s16 + a12*s22 + a13*s28 + a14*s34 + a15*s40),
            (s17, a10*s11 + a11*s17 + a12*s23 + a13*s29 + a14*s35 + a15*s41),
            # third row ...
            (s18, a20*s6  + a21*s12 + a22*s18 + a23*s24 + a24*s30 + a25*s36),
            (s19, a20*s7  + a21*s13 + a22*s19 + a23*s25 + a24*s31 + a25*s37),
            (s20, a20*s8  + a21*s14 + a22*s20 + a23*s26 + a24*s32 + a25*s38),
            (s21, a20*s9  + a21*s15 + a22*s21 + a23*s27 + a24*s33 + a25*s39),
            (s22, a20*s10 + a21*s16 + a22*s22 + a23*s28 + a24*s34 + a25*s40),
            (s23, a20*s11 + a21*s17 + a22*s23 + a23*s29 + a24*s35 + a25*s41),
            # fourth row ...
            (s24, dsum_1*s6  + d4*s12 + d5*s18 + a33*s24 + a34*s30 + a35*s36 ),
            (s25, dsum_1*s7  + d4*s13 + d5*s19 + a33*s25 + a34*s31 + a35*s37 ),
            (s26, dsum_1*s8  + d4*s14 + d5*s20 + a33*s26 + a34*s32 + a35*s38 ),
            (s27, dsum_1*s9  + d4*s15 + d5*s21 + a33*s27 + a34*s33 + a35*s39 ),
            (s28, dsum_1*s10 + d4*s16 + d5*s22 + a33*s28 + a34*s34 + a35*s40 ),
            (s29, dsum_1*s11 + d4*s17 + d5*s23 + a33*s29 + a34*s35 + a35*s41 ),
            # fifth row ...
            (s30, d4*s6  + dsum_2*s12 + d7*s18 + a43*s24 + a44*s30 + a45*s36 ),
            (s31, d4*s7  + dsum_2*s13 + d7*s19 + a43*s25 + a44*s31 + a45*s37 ),
            (s32, d4*s8  + dsum_2*s14 + d7*s20 + a43*s26 + a44*s32 + a45*s38 ),
            (s33, d4*s9  + dsum_2*s15 + d7*s21 + a43*s27 + a44*s33 + a45*s39 ),
            (s34, d4*s10 + dsum_2*s16 + d7*s22 + a43*s28 + a44*s34 + a45*s40 ),
            (s35, d4*s11 + dsum_2*s17 + d7*s23 + a43*s29 + a44*s35 + a45*s41 ),
            # sixth row ...
            (s36, d5*s6  + d7*s12 + dsum_3*s18 + a53*s24 + a54*s30 + a55*s36 ),
            (s37, d5*s7  + d7*s13 + dsum_3*s19 + a53*s25 + a54*s31 + a55*s37 ),
            (s38, d5*s8  + d7*s14 + dsum_3*s20 + a53*s26 + a54*s32 + a55*s38 ),
            (s39, d5*s9  + d7*s15 + dsum_3*s21 + a53*s27 + a54*s33 + a55*s39 ),
            (s40, d5*s10 + d7*s16 + dsum_3*s22 + a53*s28 + a54*s34 + a55*s40 ),
            (s41, d5*s11 + d7*s17 + dsum_3*s23 + a53*s29 + a54*s35 + a55*s41 ),
        ]

    if batched:
        batch_size = hy.recommended_simd_size()
        pars = np.array(batch_size * [mu]).reshape(1, -1)
        tmp_ic = np.array([tmp_ic for _ in range(batch_size)]).T

        ta = hy.taylor_adaptive_batch(ode_sys, tmp_ic, pars=pars)
    else:
        ta = hy.taylor_adaptive(ode_sys, tmp_ic, pars = [mu,])

    # construct integrator
    return ta


def gen_state_history(ta,
                      initial_state: np.ndarray[float],
                      time: float,
                      n_points: int,
                      phase: float = 0):
    """
    Generates state history for an initial condition under the CR3BP

    Args:
        ta (hy.taylor_adaptive_batch): taylor integrator for CR3BP
        initial_state: intial state of shape (6,) or shape (B, 6) 
        time: amount of time in TU to integrate for. Either float or shape (B,)
        n_points: number of grid points to return solution at.
        phase: initial phase of satellites. The satellites are first propagated forward in time by `phase * time` TU and the state history is discarded. Then the 
               satellites are propagated for another `time` TU, and this state history is what is returned by this function. Either float or shape (B,) of floats


    Returns:
        tt (np.ndarray): array of shape (B, num_points) representing the time grid the solution is evaluated at
        out (np.ndarray): array of shape (B, num_points, 6 + 6**2)

    """
    ta_b = ta.batch_size
    state_dim = 6
    stm_dim = 36 if ta.state.size == 42*ta_b else 0
    stm_ic_block = (np.tile(np.eye(state_dim).flatten(), (ta_b, 1)),) if ta.state.size == 42*ta_b else ()

    initial_state = initial_state.reshape(-1, state_dim)

    b, _ = initial_state.shape

    if b % ta_b:
        initial_state = np.vstack((initial_state,) + (b % ta_b)*(initial_state[0],))

    B, _ = initial_state.shape

    if is_not_array_like(time):
        time = np.array(B * (time,))

    if is_not_array_like(phase):
        phase = np.array(B * (phase,))
    
    tt = np.array([np.linspace(0, t, n_points) for t in time]) # (B, n_points)

    outs = np.zeros(shape=(B, n_points, state_dim + stm_dim))

    for i in range(0, B, ta_b):
        ta.state[:] = np.hstack((initial_state[i:i+ta_b],) + stm_ic_block).T
        ta.propagate_for(phase[i:i+ta_b] * time[i:i+ta_b])

        ta.set_time(ta_b * [0.])

        _, out = ta.propagate_grid(tt[i:i+ta_b].T) # (n_points, dimension, ta_b)

        outs[i:i+ta_b] = out.transpose(2, 0, 1) # (ta_b, n_points, dimension)

    outs = outs[:b]

    return tt, outs   # _, (ta_b, n_points, state_dim)

def is_not_array_like(x):
    return not isinstance(x, (np.ndarray, list, tuple))

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
    
    initial_state = np.tile(initial_state, (7, 1))
    time_ = 3.225
    n_points = 215
    phase = 0

    import time
    start = time.perf_counter()

    _, out = gen_state_history(ta=ta,
                            initial_state=initial_state,
                            time=time_,
                            n_points=n_points,
                            phase=phase)
    end = time.perf_counter()

    print(f"Elapsed time: {end - start:.6f} seconds")

    print(out.shape)

