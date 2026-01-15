import numpy as np
import gurobipy as gp
from gurobipy import GRB
from ortools.sat.python import cp_model
import pandas as pd
from typing import Optional

def solve_assignment_problem_cpsat(weights : np.ndarray[float],
                                   opt_type: Optional[str] = "max"):
    
    """
    Use google or-tools CP-SAT solver to solve the assignment problem given a weight matrix describing the profits/cost of assigning
    a servicer to a customer

    Args:
        weights (np.ndarray) : a 2D array. weights[i, j] describes the profit/cost of assigning 
        servicer i to customer j
        opt_type (str) : either "max", so weights are interpreted as profits, or "min", so weights
        are interpreted as costs
    
    Returns:
        assignment (np.ndarray) : a 2D assignment array. x[i, j] = 1 if servicer i assigned to customer j
        objective (float) : optimal objective value
    """

    model = cp_model.CpModel()

    x = model.new_bool_var_series(name="x", index=pd.Index(np.arange(weights.size)))

    num_servicers, num_customers = weights.shape

    # Constraints
    # Each servicer is assigned to at most one customer.
    for i in range(num_servicers):
        index = pd.Index([i * num_customers + j for j in range(num_customers)])
        model.add_at_most_one(x[index])

    # Each customer is assigned to exactly one servicer.
    for j in range(num_customers):
        index = pd.Index([i * num_customers + j for i in range(num_servicers)])
        model.add_exactly_one(x[index])

    if opt_type == "max":
        model.maximize(weights.flatten().dot(x))
    else:
        model.minimize(weights.flatten().dot(x))
    
    # Solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    # verify solution.
    if status == cp_model.OPTIMAL:
        pass
    elif status == cp_model.INFEASIBLE:
        raise RuntimeError("No solution found")
    else:
        raise RuntimeError("Solver error")

    return solver.boolean_values(x).values.reshape(weights.shape).astype(int), solver.objective_value


def solve_assignment_problem_time_expanded(weights : np.ndarray[float],
                                           assignment_penalty_lambda: Optional[np.ndarray[float]] = 0.0,
                                           opt_type: Optional[str] = "max"):
    """
    Solve the assignment problem given a weight matrix describing the profits/cost of assigning
    a servicer to a customer, with an additional penalty for assigning servicers/customers.

    Args:
        weights (np.ndarray) : a 2D array. weights[k, i, j] describes the profit/cost of assigning a servicer i to customer j at time k
        assignment_penalty_lambda (np.ndarray) : a 2D array. assignment_penalty_lambda[k, i, j] describes the penalty for assigning servicer i to customer j at time k
        opt_type (str) : either "max", so weights are interpreted as profits, or "min"

    Returns:
        assignment (np.ndarray) : a 3D assignment array. x[k, i, j] = 1 if servicer i assigned to customer j at time k
        objective (float) : optimal objective value
    """

    n_timesteps, n_servicers, n_customers = weights.shape

    # standardize weights into [0, 1] (in order to ensure coefficients from logdet information objective become positive)
    weights_norm = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

    if opt_type == "min":
        weights_norm = 1 / (weights_norm + 1e-6)  # avoid division by zero

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    m = gp.Model("time expanded assignment problem", env=env)

    OPT = GRB.MAXIMIZE

    # Silence model output
    m.Params.LogToConsole = 0

    # Create variables
    x = m.addMVar(shape=weights.shape, vtype=GRB.BINARY, name="x")

    # Set objective
    m.setObjective(((weights_norm - assignment_penalty_lambda) * x).sum(), OPT)

    # each servicer can service at most 1 customer
    # ∑_{j ∈ Customers} x_ij <= 1,  ∀ i ∈ Servicers
    m.addConstr(x.sum(axis=2) <= 1)

    # each customer gets at most one servicer
    # ∑_{i ∈ Servicers} x_ij <= 1,  ∀ j ∈ Customers
    m.addConstr(x.sum(axis=1) <= 1)

    m.optimize()

    if m.status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Model was not solved, model status: {m.status}")

    np.rint(x.X, out=x.X)
    assignment = x.X.astype(int)
    # objective = m.getObjective().getValue()
    objective = ((weights * assignment).sum())

    return assignment , objective



def solve_assignment_problem(weights : np.ndarray[float],
                             opt_type: Optional[str] = "max"):
    """
    Solve the assignment problem given a weight matrix describing the profits/cost of assigning 
    a servicer to a customer

    Args:
        weights (np.ndarray) : a 2D array. weights[i, j] describes the profit/cost of assigning 
        servicer i to customer j
        opt_type (str) : either "max", so weights are interpreted as profits, or "min", so weights
        are interpreted as costs
    Returns:
        assignment (np.ndarray) : a 2D assignment array. x[i, j] = 1 if servicer i assigned to customer j
        objective (float) : optimal objective value
    """

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    n_servicers, n_customers = weights.shape

    m = gp.Model("assignment problem", env=env)

    OPT = GRB.MAXIMIZE if opt_type == "max" else GRB.MINIMIZE

    # Silence model output
    m.Params.LogToConsole = 0

    # Create variables
    x = m.addMVar(shape=weights.shape, vtype=GRB.BINARY, name="x")

    # Set objective
    m.setObjective((weights * x).sum(), OPT)

    if n_customers == n_servicers:
        # each servicer can service exactly 1 customer
        # ∑_{j ∈ Customers} x_ij = 1,  ∀ i ∈ Servicers
        m.addConstr(x.sum(axis=1) == 1)

        # each customer gets exactly one servicer
        # ∑_{i ∈ Servicers} x_ij = 1,  ∀ j ∈ Customers
        m.addConstr(x.sum(axis=0) == 1)

    elif n_customers < n_servicers:
        m.addConstr(x.sum(axis=0) == 1)
        m.addConstr(x.sum(axis=1) <= 1)

    else:
        m.addConstr(x.sum(axis=0) <= 1)
        m.addConstr(x.sum(axis=1) == 1)

    m.optimize()

    if m.status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Model was not solved, model status: {m.status}")

    np.rint(x.X, out=x.X)
    assignment = x.X.astype(int)
    objective = m.getObjective().getValue()

    return assignment , objective

if __name__ == "__main__":

    num_servicers = 2
    num_customers = 3
    n_timesteps = 4

    print(f"Testing with {num_servicers} servicers and {num_customers} customers for {n_timesteps} timesteps")

    np.random.seed(1)
    weights = np.random.uniform(low = 0, high=1, size=(n_timesteps, num_servicers, num_customers))
    assignment_penalty_lambda = 0.1

    x, obj = solve_assignment_problem_time_expanded(weights, assignment_penalty_lambda=assignment_penalty_lambda, opt_type="max")

    # print(f"Optimal objective : {obj}\nAssignment:\n{x}")
    # print(f"Average number of assignments per observer per timestep: {x.sum() / n_timesteps / num_servicers}")

    res = []
    for lamda in np.linspace(0, 1, 100):
        assignment_penalty_lambda = lamda * np.ones_like(weights)
        x, obj = solve_assignment_problem_time_expanded(weights, assignment_penalty_lambda=assignment_penalty_lambda, opt_type="max")
        avg_assignments = x.sum() / n_timesteps / num_servicers
        obj_per_assignment = ((weights * x).sum()) / x.sum() if x.sum() > 0 else 0
        res.append((lamda, obj_per_assignment, avg_assignments))
        # print(f"Lambda: {lamda:.2f}, Optimal objective: {obj:.4f}, Average assignments per servicer per timestep: {avg_assignments:.4f}")


    import matplotlib.pyplot as plt

    res = np.array(res)
    plt.figure(figsize=(12, 5))

    plt.plot(res[:, 0], res[:, 2], label="Average Assignments per Servicer per Timestep")
    plt.plot(res[:, 0], res[:, 1] , label="Objective per Assignment")
    plt.xlabel("Assignment Penalty Lambda")
    plt.ylabel("Average Assignments per Servicer per Timestep")
    plt.title("Effect of Assignment Penalty on Average Assignments")
    plt.legend()
    plt.show()

