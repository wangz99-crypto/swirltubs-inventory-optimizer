import pandas as pd
from ortools.linear_solver import pywraplp


def solve_milp_max_total_net_savings(df2: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """
    Maximize total net savings:
      maximize sum_i x_i * net_savings_i
      s.t. sum_i x_i * size_i <= capacity, x_i in {0,1}

    Returns a dataframe with a 'decision' column (0/1) and total_cost_item.
    """
    d = df2.reset_index(drop=True).copy()

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver not available.")

    x = [solver.BoolVar(f"x_{i}") for i in range(len(d))]

    # Capacity constraint
    solver.Add(
        solver.Sum(x[i] * float(d.loc[i, "size"]) for i in range(len(d))) <= float(capacity)
    )

    # Objective: maximize total net savings
    solver.Maximize(
        solver.Sum(x[i] * float(d.loc[i, "net_savings"]) for i in range(len(d)))
    )

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("MILP did not find a feasible solution.")

    d["decision"] = [1 if x[i].solution_value() > 0.5 else 0 for i in range(len(d))]
    # per-item total cost (same as Excel logic)
    d["total_cost_item"] = d["decision"] * d["holding_cost"] + (1 - d["decision"]) * d["revisit_cost"]
    return d


# (Optional) keep your old function for backward compatibility
def solve_milp_min_total_cost(df2: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """
    Minimize sum( x_i*holding_cost + (1-x_i)*revisit_cost )
    Equivalent to maximizing savings vs a baseline, but kept for compatibility.
    """
    d = df2.reset_index(drop=True).copy()

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver not available.")

    x = [solver.BoolVar(f"x_{i}") for i in range(len(d))]

    solver.Add(solver.Sum(x[i] * float(d.loc[i, "size"]) for i in range(len(d))) <= float(capacity))

    obj_terms = []
    for i in range(len(d)):
        hold = float(d.loc[i, "holding_cost"])
        rev = float(d.loc[i, "revisit_cost"])
        obj_terms.append(x[i] * hold + (1 - x[i]) * rev)

    solver.Minimize(solver.Sum(obj_terms))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("MILP did not find a feasible solution.")

    d["decision"] = [1 if x[i].solution_value() > 0.5 else 0 for i in range(len(d))]
    d["total_cost_item"] = d["decision"] * d["holding_cost"] + (1 - d["decision"]) * d["revisit_cost"]
    return d