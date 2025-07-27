import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Optional

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

    num_servicers = 5
    num_customers = 4

    print(f"Testing with {num_servicers} servicers and {num_customers} customers...")

    weights = np.random.rand(num_servicers, num_customers)

    x, obj = solve_assignment_problem(weights)

    print(f"Optimal objective : {obj}\nAssignment:\n{x}")
