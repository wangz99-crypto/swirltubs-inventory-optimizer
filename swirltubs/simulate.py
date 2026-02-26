import numpy as np
import pandas as pd
from .policy import kpis


def _infer_revisit_cost_per(df2: pd.DataFrame, fallback: float = 25.0) -> float:
    """
    Estimate revisit_cost_per from df2 (revisit_cost / annual_use) using median of valid rows.
    """
    x = (df2["revisit_cost"] / df2["annual_use"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return float(fallback)
    v = float(x.median())
    return v if np.isfinite(v) else float(fallback)


def run_monte_carlo_fixed_portfolio_poisson(
    df2: pd.DataFrame,
    portfolio: pd.DataFrame,
    n_sims: int = 200,
    seed: int = 42,
    lam_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Poisson demand simulation (count-based):
      annual_use_i_sim ~ Poisson(lambda = annual_use_i * lam_scale)

    We keep the same stocking decisions (portfolio), recompute revisit-related cost,
    and evaluate KPIs across scenarios.
    """
    rng = np.random.default_rng(seed)

    base = df2.copy()
    dec = portfolio[["part", "decision"]].copy()
    base = base.merge(dec, on="part", how="left")
    base["decision"] = base["decision"].fillna(0).astype(int)

    revisit_cost_per = _infer_revisit_cost_per(df2, fallback=25.0)

    lam = (base["annual_use"].clip(lower=0).to_numpy()) * float(lam_scale)

    rows = []
    for s in range(int(n_sims)):
        demand_sim = rng.poisson(lam=lam)

        sim = base.copy()
        sim["annual_use"] = demand_sim
        sim["revisit_cost"] = revisit_cost_per * sim["annual_use"]
        sim["net_savings"] = sim["revisit_cost"] - sim["holding_cost"]
        sim["net_savings_per_cuft"] = sim["net_savings"] / sim["size"]
        sim["annual_use_per_cuft"] = sim["annual_use"] / sim["size"]

        # Excel-style cost per item based on decision
        sim["total_cost_item"] = sim["decision"] * sim["holding_cost"] + (1 - sim["decision"]) * sim["revisit_cost"]

        m = kpis(sim)
        rows.append({
            "sim": s + 1,
            "lam_scale": float(lam_scale),
            "total_cost": m["total_cost"],
            "fix_first": m["fix_first"],
            "total_net_savings": m["total_net_savings"],
        })

    return pd.DataFrame(rows)


def run_poisson_risk_sweep(
    df2: pd.DataFrame,
    portfolio: pd.DataFrame,
    risk_levels=(0.10, 0.20, 0.30, 0.40, 0.50),
    n_sims: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Risk level is interpreted as uncertainty on lambda:
      lambda_sim = annual_use * (1 + eps), where eps ~ Uniform(-risk, +risk)

    We implement it by sampling a scale per simulation:
      lam_scale ~ Uniform(1-risk, 1+risk)
    and then Poisson sampling on that lambda.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for i, r in enumerate(risk_levels):
        r = float(r)
        # run many sims with random lambda scales in [1-r, 1+r]
        sims_all = []
        for s in range(int(n_sims)):
            lam_scale = rng.uniform(1.0 - r, 1.0 + r)
            sims_all.append(
                run_monte_carlo_fixed_portfolio_poisson(
                    df2, portfolio, n_sims=1, seed=seed + 1000 * i + s, lam_scale=lam_scale
                ).iloc[0].to_dict()
            )
        sims = pd.DataFrame(sims_all)

        rows.append({
            "risk": r,
            "cost_mean": sims["total_cost"].mean(),
            "cost_std": sims["total_cost"].std(ddof=1),
            "fix_mean": sims["fix_first"].mean(),
            "fix_std": sims["fix_first"].std(ddof=1),
        })

    return pd.DataFrame(rows)