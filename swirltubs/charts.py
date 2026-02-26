# swirltubs/charts.py
from __future__ import annotations

import pandas as pd
import plotly.express as px


def summarize_mc(df_mc: pd.DataFrame) -> pd.DataFrame:
    """
    df_mc columns expected (at least):
      sim, risk_pct, total_cost, fix_first, space_used, parts_selected

    Return:
      risk_pct + mean/std for each metric
    """
    if df_mc.empty:
        return pd.DataFrame()

    g = df_mc.groupby("risk_pct", as_index=False).agg(
        total_cost_mean=("total_cost", "mean"),
        total_cost_std=("total_cost", "std"),
        total_net_savings_mean=("total_net_savings", "mean"),
        total_net_savings_std=("total_net_savings", "std"),
        fix_first_mean=("fix_first", "mean"),
        fix_first_std=("fix_first", "std"),
        space_used_mean=("space_used", "mean"),
        space_used_std=("space_used", "std"),
        parts_selected_mean=("parts_selected", "mean"),
        parts_selected_std=("parts_selected", "std"),
    )
    g = g.sort_values("risk_pct").reset_index(drop=True)
    return g


def _fmt_pct_axis(fig):
    fig.update_yaxes(tickformat=".0%")
    return fig


def plot_sensitivity_line(
    df_summary: pd.DataFrame,
    y_col: str,
    title: str,
    y_is_percent: bool = False,
):
    fig = px.line(
        df_summary,
        x="risk_pct",
        y=y_col,
        markers=True,
        title=title,
    )
    fig.update_xaxes(title="Fluctuation rate", tickformat=".0%")
    fig.update_yaxes(title=y_col.replace("_", " ").title())
    if y_is_percent:
        _fmt_pct_axis(fig)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_sim_series(df_mc_one: pd.DataFrame, y: str, title: str, y_is_percent: bool = False):
    fig = px.line(
        df_mc_one.sort_values("sim"),
        x="sim",
        y=y,
        title=title,
    )
    fig.update_xaxes(title="Simulation run")
    fig.update_yaxes(title=y.replace("_", " ").title())
    if y_is_percent:
        _fmt_pct_axis(fig)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_sim_hist(df_mc_one: pd.DataFrame, x: str, title: str, x_is_percent: bool = False, nbins: int = 16):
    fig = px.histogram(
        df_mc_one,
        x=x,
        nbins=nbins,
        title=title,
    )
    fig.update_xaxes(title=x.replace("_", " ").title())
    if x_is_percent:
        fig.update_xaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


# ----------------------------
# Backwards-compatible names (your earlier import error)
# ----------------------------
def plot_sim_cost_series(df_mc_one: pd.DataFrame, title: str = "Total cost across simulations"):
    return plot_sim_series(df_mc_one, y="total_cost", title=title, y_is_percent=False)


def plot_sim_fix_series(df_mc_one: pd.DataFrame, title: str = "Fix-it-first across simulations"):
    return plot_sim_series(df_mc_one, y="fix_first", title=title, y_is_percent=True)
