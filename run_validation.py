from valpy.model import ValidationModel
import rudolfpy as rd
import numpy as np
import matplotlib.pyplot as plt

from src.Part3.dynamics import gen_state_history, build_taylor_cr3bp

plt.rcParams.update({'font.size': 14})

# seed 0 gives okay result
# np.random.seed(100)


# read .npz file
config = np.load("exp7.npz")


# CR3BP dynamics parameters
mu = config["mu"].item()
LU = config["LU"].item()
TU = config["TU"].item()
VU = LU / TU

# measurement model parameters
sigma = config["sigma"].item()
t_expose = config["t_expose"].item()



# random control
n = 2  # size of permutation matrix (choose your n)
num_slices = 215

U = np.zeros((n, n, num_slices), dtype=int)


for i in range(num_slices):
    perm = np.random.permutation(n)     # e.g. [2,0,3,1,...]
    U[perm, np.arange(n), i] = 1        # fill the permutation matrix

U = np.concatenate([U, U], axis=-1)

u = config["u"].transpose(1, 2, 0)
u = np.concatenate([u, u], axis=-1)

# slicing
slice_factor = 1
u = u[..., ::slice_factor]
U = U[..., ::slice_factor]

optimal_phases = config["optimal_observer_phases"]
initial_state_observer = config["initial_state_observer"]
initial_state_target = config["initial_state_target"]
initial_state_target_phases = config["initial_state_target_phases"]

ta = build_taylor_cr3bp(mu=mu, stm=False, batched=True)
_, states_x = gen_state_history(ta=ta,
                        initial_state=initial_state_observer,
                        time=1.0,
                        n_points=10,
                        phase=optimal_phases)
_, states_y = gen_state_history(ta=ta,
                        initial_state=initial_state_target,
                        time=1.0,
                        n_points=10,
                        phase=initial_state_target_phases)


## ground truth initial conditions for targets and observers
groundtruth_targets = states_y[:, 0, :]
groundtruth_observers = states_x[:, 0, :]


# positional uncertainty (is used to compute initial covariance of target and observer state)
sigma_r0 = 100 / LU * 10
sigma_v0 = 0.01 / VU * 10


# estimates for (x0, P0) for observer and targets
estimate_observers = groundtruth_observers + np.array([sigma_r0]*3 + [sigma_v0]*3) * np.random.normal(0, 1, size=groundtruth_observers.shape)
estimate_targets   = groundtruth_targets   + np.array([sigma_r0]*3 + [sigma_v0]*3) * np.random.normal(0, 1, size=groundtruth_targets.shape)
P0_observers = np.tile(np.diag([sigma_r0]*3 + [sigma_v0]*3)**2, (groundtruth_observers.shape[0], 1, 1))
P0_targets   = np.tile(np.diag([sigma_r0]*3 + [sigma_v0]*3)**2, (groundtruth_targets.shape[0], 1, 1))

# create dynamics model
dynamics = rd.DynamicsCR3BP(mu = mu, LU=LU, TU=TU, method='DOP853', rtol = 1e-12, atol = 1e-12)
# create measurement model
meas_model = rd.MeasurementAngleAngleRate()
# process noise
params_Q = [1e-7,]
# create EKF object
filter = rd.UnscentedKalmanFilter(dynamics, meas_model,
                                 func_process_noise = rd.unbiased_random_process_3dof,
                                 params_Q = params_Q,)
# timestep of simulation
timestep = 0.015 * slice_factor



vm = ValidationModel(
                     groundtruth_observers=groundtruth_observers,
                     groundtruth_targets=groundtruth_targets,
                     estimate_observers=estimate_observers,
                     estimate_targets=estimate_targets,
                     P0_observers=P0_observers,
                     P0_targets=P0_targets,
                     filter=filter,
                     timestep=timestep)

target_hist, _ = vm.run(u=u,filter_observers=False, meas_model_sigma = sigma, dt_exposure = t_expose)
target_hist_rand, _ = vm.run(u=U,filter_observers=False, meas_model_sigma = sigma, dt_exposure = t_expose)




##### PLOTTING #####

state_to_var = {0:"x", 1:"y", 2:"z", 3:"vx", 4:"vy", 5:"vz"}

state_idx = 0
target_idx = 0

metric = "determinant"  # "logdet" or "trace"

target_hist_gt = vm._get_target_history_groundtruth(u, target_idx)

t = np.linspace(0, u.shape[2] * timestep, len(target_hist[target_idx]))

if metric == "determinant":

    tr = np.array([np.linalg.slogdet(item[-1][:4, :4])[1] for item in target_hist[target_idx]])
    tr_random = np.array([np.linalg.slogdet(item[-1][:4, :4])[1] for item in target_hist_rand[target_idx]])

    tr2 = np.array([np.linalg.slogdet(item[-1][:4, :4])[1] for item in target_hist[target_idx+1]])
    tr_random2 = np.array([np.linalg.slogdet(item[-1][:4, :4])[1] for item in target_hist_rand[target_idx+1]])

elif metric == "trace":

    tr  = np.array([np.log(np.trace(item[-1][:4, :4])) 
                    for item in target_hist[target_idx]])

    tr_random = np.array([np.log(np.trace(item[-1][:4, :4])) 
                        for item in target_hist_rand[target_idx]])

    tr2 = np.array([np.log(np.trace(item[-1][:4, :4])) 
                    for item in target_hist[target_idx+1]])

    tr_random2 = np.array([np.log(np.trace(item[-1][:4, :4])) 
                        for item in target_hist_rand[target_idx+1]])
    
else:
    raise ValueError("Invalid metric")

states_x = np.array([item[0][state_idx] for item in target_hist[target_idx]])
states_true = np.array([item[0][state_idx] for item in target_hist_gt])
plt.plot(t, states_x)
plt.plot(t, states_true)
plt.xlabel("time (TU)")
plt.ylabel(f"state component {state_to_var[state_idx]}")
plt.grid(True)

plt.figure()

plt.title(f"Target 1 log-{metric}\nCovariance Evolution")
plt.plot(t, tr, label="opt")
plt.plot(t, tr_random, label="random")
plt.xlabel("time (TU)")
plt.ylabel(f"log-{metric} covariance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.figure()

plt.title(f"Target 2 log-{metric}\nCovariance Evolution")
plt.plot(t, tr2, label="opt")
plt.plot(t, tr_random2, label="random")
plt.xlabel("time (TU)")
plt.ylabel(f"log-{metric} covariance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# print(u)
# print(U)