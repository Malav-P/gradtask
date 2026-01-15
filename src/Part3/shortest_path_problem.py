import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_shortest_path(W, source, target):
    n, _ = W.shape # (N, N)
    

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    # Create model
    m = gp.Model("shortest_path", env=env)
    m.Params.LogToConsole = 0  # Silence output
    
    # Binary decision variables x[i,j] = 1 if edge i->j is used
    x = m.addMVar(shape=W.shape, vtype=GRB.BINARY, name="x")
    
    # Objective: minimize total cost
    m.setObjective((W*x).sum(), GRB.MINIMIZE)
    
    # Flow conservation constraints
    for i in range(n):
        if i == source:
            m.addConstr(gp.quicksum(x[i,j] for j in range(n)) - gp.quicksum(x[j,i] for j in range(n)) == 1)
        elif i == target:
            m.addConstr(gp.quicksum(x[i,j] for j in range(n)) - gp.quicksum(x[j,i] for j in range(n)) == -1)
        else:
            m.addConstr(gp.quicksum(x[i,j] for j in range(n)) - gp.quicksum(x[j,i] for j in range(n)) == 0)
    
    # Solve
    m.optimize()
    
    # Extract path
    path = []
    # if m.status == GRB.OPTIMAL:
    #     current = source
    #     while current != target:
    #         for j in range(n):
    #             if x[current,j].X > 0.5:
    #                 path.append((current, j))
    #                 current = j
    #                 break

    np.rint(x.X, out=x.X)
    x_var = x.X.astype(int)

    return x_var, path, m.objVal if m.status == GRB.OPTIMAL else None

def solve_shortest_path_time_expanded(W, source, target):
    T, n, _ = W.shape # (T, N, N)
    

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    # Create model
    m = gp.Model("shortest_path", env=env)
    m.Params.LogToConsole = 0  # Silence output
    
    # Binary decision variables x[i,j] = 1 if edge i->j is used
    x = m.addMVar(shape=W.shape, vtype=GRB.BINARY, name="x")
    
    # Objective: minimize total cost
    m.setObjective((W*x).sum(), GRB.MINIMIZE)
    
    # Flow conservation constraints
    for i in range(n):
        if i == source:
            m.addConstr(gp.quicksum(x[:, i,j] for j in range(n)) - gp.quicksum(x[:, j,i] for j in range(n)) == 1)
        elif i == target:
            m.addConstr(gp.quicksum(x[:, i,j] for j in range(n)) - gp.quicksum(x[:, j,i] for j in range(n)) == -1)
        else:
            m.addConstr(gp.quicksum(x[:, i,j] for j in range(n)) - gp.quicksum(x[:, j,i] for j in range(n)) == 0)
    
    # Solve
    m.optimize()
    
    # Extract path
    path = []
    # if m.status == GRB.OPTIMAL:
    #     current = source
    #     while current != target:
    #         for j in range(n):
    #             if x[current,j].X > 0.5:
    #                 path.append((current, j))
    #                 current = j
    #                 break

    np.rint(x.X, out=x.X)
    x_var = x.X.astype(int)

    return x_var, path, m.objVal if m.status == GRB.OPTIMAL else None


if __name__ == "__main__":
    from src.Part3.dynamics import gen_state_history, build_taylor_cr3bp
    from src.Part3.constants import CR3BP_MU
    from matplotlib import pyplot as plt

    time_ = 3.225
    n_points = 215
    n_satellites = 2

    # Parameters
    A = 1.0        # amplitude
    mu = 0.0       # mean
    sigma = 1.0    # standard deviation

    # Generate x values
    x = np.linspace(-5, 5, n_points)

    # Compute Gaussian values
    weights = A * np.exp(-((x + 3 - mu)**2) / (2 * sigma**2))

    ta = build_taylor_cr3bp(mu=CR3BP_MU, stm=False)

    initial_state_y = np.array([
                        0.8027692908754149,
                        0.0,
                        0.0,
                        -1.1309830924549648e-14,
                        0.33765564334938736,
                        0.0
                    ])
    mean_cost = []
    mean_cost_gauss = []
    candidiates = np.linspace(0.01, 0.99, 11)

    for phase_x in candidiates:
    
        _, states_y = gen_state_history(ta=ta,
                                initial_state=np.tile(initial_state_y, (n_satellites, 1)),
                                time=time_,
                                n_points=n_points,
                                phase=(0, phase_x))
        

        W = np.zeros((n_satellites + 2, n_satellites + 2))

        objs = []
        for t in range(n_points):
            for i in range(n_satellites+2):
                if i == 0:
                    sati = np.array([-CR3BP_MU,0,0])
                elif i == n_satellites + 1:
                    sati = np.array([1-CR3BP_MU,0,0])
                else:
                    sati = states_y[i-1, t, :3]

                for j in range(n_satellites+2):
                    
                    if i == j:
                        W[i,j] = 1000
                        continue

                    if j == 0:
                        satj = np.array([-CR3BP_MU,0,0])
                    elif j == n_satellites + 1:
                        satj = np.array([1-CR3BP_MU,0,0])
                    else:
                        satj = states_y[j-1, t, :3]

                    W[i,j] = (sati - satj).T @ (sati - satj)

            x_var, path, obj = solve_shortest_path(W, 0, n_satellites + 1)
            objs.append(obj)

        mean_cost_gauss.append(np.average(objs, weights=weights))
        mean_cost.append(np.average(objs, weights=None))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # Left: unweighted mean
    axes[0].plot(candidiates, mean_cost, marker='o')
    axes[0].set_xlabel(r"$\phi$")
    axes[0].set_ylabel(r"$\mathrm{Agg}\left(\text{LP}_{t}\right)$")
    axes[0].set_title("Mean Shortest Path Cost")
    axes[0].grid(True)

    # Right: weighted mean (Gaussian or whatever your weights are)
    axes[1].plot(candidiates, mean_cost_gauss, marker='o')
    axes[1].set_xlabel(r"$\phi$")
    axes[1].set_title("Weighted Mean Shortest Path Cost")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


