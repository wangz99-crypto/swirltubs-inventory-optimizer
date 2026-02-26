import pandas as pd


def compute_fields(df: pd.DataFrame, holding_rate: float = 0.25, revisit_cost_per: float = 25.0) -> pd.DataFrame:
    """
    Adds economics fields used across heuristics, MILP, and simulation.

    Required inputs:
      cost, annual_use, size

    Definitions:
      holding_cost = holding_rate * cost
      revisit_cost = revisit_cost_per * annual_use
      net_savings  = revisit_cost - holding_cost
    """
    out = df.copy()
    out["holding_cost"] = holding_rate * out["cost"]
    out["revisit_cost"] = revisit_cost_per * out["annual_use"]
    out["net_savings"] = out["revisit_cost"] - out["holding_cost"]
    out["net_savings_per_cuft"] = out["net_savings"] / out["size"]
    out["annual_use_per_cuft"] = out["annual_use"] / out["size"]
    return out


def apply_decision_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure total_cost_item exists based on the per-item cost rule."""
    out = df.copy()
    if "decision" not in out.columns:
        raise ValueError("apply_decision_costs expects a 'decision' column.")
    out["total_cost_item"] = out["decision"] * out["holding_cost"] + (1 - out["decision"]) * out["revisit_cost"]
    return out


def greedy_fill(df: pd.DataFrame, sort_col: str, capacity: float, ascending: bool) -> pd.DataFrame:
    """
    Fast selection rule:
    sort by a metric, then take the prefix until cumulative space <= capacity.
    """
    out = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True).copy()
    out["cumulative_space"] = out["size"].cumsum()
    out["decision"] = (out["cumulative_space"] <= capacity).astype(int)
    out = apply_decision_costs(out)
    return out


def kpis(df_decided: pd.DataFrame) -> dict:
    d = df_decided

    if "total_cost_item" not in d.columns:
        d["total_cost_item"] = d["decision"] * d["holding_cost"] + (1 - d["decision"]) * d["revisit_cost"]

    total_cost = float(d["total_cost_item"].sum())
    space_used = float((d["decision"] * d["size"]).sum())
    parts_selected = int(d["decision"].sum())

    demand_total = float(d["annual_use"].sum()) if d["annual_use"].sum() else 0.0
    fix_first = float((d["decision"] * d["annual_use"]).sum() / demand_total) if demand_total else 0.0

    total_net_savings = float((d["decision"] * d["net_savings"]).sum()) if "net_savings" in d.columns else 0.0
    roi_per_cuft = (total_net_savings / space_used) if space_used > 0 else 0.0

    return {
        "total_cost": total_cost,
        "space_used": space_used,
        "parts_selected": parts_selected,
        "fix_first": fix_first,
        "total_net_savings": total_net_savings,
        "roi_per_cuft": roi_per_cuft,
    }


def run_all_policies(df2: pd.DataFrame, capacity: float) -> dict:
    """
    Alternative fast rules (not guaranteed optimal).
    Names are written for non-technical users.
    """
    return {
        "Fast Rule – High ROI per Space": greedy_fill(df2, "net_savings_per_cuft", capacity, ascending=False),
        "Fast Rule – Highest Total Savings": greedy_fill(df2, "net_savings", capacity, ascending=False),
        "Fast Rule – Best Coverage (Most Used)": greedy_fill(df2, "annual_use", capacity, ascending=False),
        "Fast Rule – Smallest Parts First": greedy_fill(df2, "size", capacity, ascending=True),
        "Fast Rule – Lowest Cost First": greedy_fill(df2, "cost", capacity, ascending=True),
    }