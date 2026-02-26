import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from swirltubs.io import load_part1
from swirltubs.policy import compute_fields, run_all_policies, kpis
from swirltubs.optimize_milp import solve_milp_max_total_net_savings
from swirltubs.simulate import (
    run_monte_carlo_fixed_portfolio_poisson,
    run_poisson_risk_sweep,
)

st.set_page_config(page_title="Swirltubs | Van Inventory Decision Tool", layout="wide")

# ✅ change to your full dataset file path
DATA_PART1 = "data/part1_model.xlsx"


def money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def pct(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return str(x)


@st.cache_data(show_spinner=False)
def cached_load_full(path: str) -> pd.DataFrame:
    # ✅ only full mode now
    return load_part1(path, mode="full")


@st.cache_data(show_spinner=False)
def cached_compute(df: pd.DataFrame, holding_rate: float, revisit_cost_per: float) -> pd.DataFrame:
    return compute_fields(df, holding_rate=holding_rate, revisit_cost_per=revisit_cost_per)


@st.cache_data(show_spinner=False)
def cached_milp(df2: pd.DataFrame, capacity: float) -> pd.DataFrame:
    return solve_milp_max_total_net_savings(df2, capacity)


@st.cache_data(show_spinner=False)
def cached_heuristics(df2: pd.DataFrame, capacity: float) -> dict:
    return run_all_policies(df2, capacity)


st.title("Service Van Inventory Decision Tool")
st.caption("Goal: maximize total annual net savings under a van storage capacity constraint.")


tab_overview, tab_setup, tab_rec, tab_alt, tab_risk = st.tabs([
    "Overview",
    "Decision Setup",
    "Recommended Plan",
    "Alternatives",
    "Demand Uncertainty"
])


with tab_overview:
    st.subheader("Business context")
    st.markdown(
        """
Swirltubs manages a large network of field technicians. Each technician stocks parts in a service van.

- Stocking more parts increases **inventory holding cost** (assume 25% of part cost per year).
- Stocking too few parts increases **revisit cost** when a repair cannot be completed on the first visit.
- Vans have a limited storage capacity (e.g., 500 cubic feet).

This tool recommends which parts to stock to maximize total annual **net savings**.
"""
    )
    st.info("Go to **Decision Setup** → click **Generate Recommended Plan**.")


with tab_setup:
    st.subheader("Decision inputs")

    c1, c2 = st.columns([1.15, 0.85])

    with c1:
        capacity = st.slider(
            "Available Van Storage Capacity (cubic feet)",
            min_value=1.0, max_value=500.0, value=500.0, step=5.0
        )
        holding_rate = st.slider(
            "Annual Holding Rate",
            min_value=0.0, max_value=0.60, value=0.25, step=0.01
        )
        revisit_cost_per = st.slider(
            "Revisit Cost per Visit ($)",
            min_value=0.0, max_value=200.0, value=25.0, step=1.0
        )

    with c2:
        st.markdown("### Demand uncertainty settings")
        st.caption("Used to stress-test the plan under fluctuating annual part demand.")

        n_sims = st.slider(
            "Simulation runs",
            50, 800, 200, 50,
            help="More runs = smoother results, but takes longer."
        )

        risk = st.select_slider(
            "Demand uncertainty level",
            options=[0.10, 0.20, 0.30, 0.40, 0.50],
            value=0.20,
            help=(
                "Controls how much annual demand may fluctuate around the historical average. "
                "Higher values mean more volatility (more uncertainty)."
            )
        )

        seed = st.number_input(
            "Random seed",
            min_value=0, value=42, step=1,
            help="Keep it fixed for reproducible results."
        )

    st.divider()

    if st.button("Generate Recommended Plan", type="primary", use_container_width=True):
        with st.spinner("Computing..."):
            df = cached_load_full(DATA_PART1)
            df2 = cached_compute(df, float(holding_rate), float(revisit_cost_per))

            recommended = cached_milp(df2, float(capacity))
            heuristics = cached_heuristics(df2, float(capacity))

        st.session_state["df2"] = df2
        st.session_state["recommended"] = recommended
        st.session_state["heuristics"] = heuristics
        st.session_state["inputs"] = {
            "capacity": float(capacity),
            "holding_rate": float(holding_rate),
            "revisit_cost_per": float(revisit_cost_per),
            "n_sims": int(n_sims),
            "risk": float(risk),
            "seed": int(seed),
        }
        st.success("Done. Go to Recommended plan and Demand Uncertainty.")


def require_solution():
    if "recommended" not in st.session_state:
        st.warning("Please go to Decision Setup and click **Generate Recommended Plan** first.")
        st.stop()


SHOW_COLS = [
    "part", "size", "cost", "annual_use",
    "holding_cost", "revisit_cost", "net_savings",
    "net_savings_per_cuft", "decision"
]


with tab_rec:
    require_solution()
    st.subheader("Recommended (Optimal) — Max Total Net Savings")
    st.caption(
    """
This plan is generated using an optimization model that evaluates all feasible stocking combinations 
under the van capacity constraint and selects the mix of parts that maximizes total annual net savings.
"""
)

    rec = st.session_state["recommended"]
    m = kpis(rec)

    a, b, c, d, e = st.columns(5)
    a.metric("Total Net Savings", money(m["total_net_savings"]))
    b.metric("Total Cost", money(m["total_cost"]))
    c.metric("Fix-it-First (Demand Covered)", pct(m["fix_first"]))
    d.metric("Capacity Used", f"{m['space_used']:.2f} cu ft")
    e.metric("ROI per cu ft", money(m["roi_per_cuft"]))

    chosen = rec.loc[rec["decision"] == 1, SHOW_COLS].sort_values("net_savings", ascending=False)
    st.markdown("### Selected parts")
    st.dataframe(chosen.head(300), use_container_width=True)

    st.download_button(
        "Download selected parts (CSV)",
        data=chosen.to_csv(index=False).encode("utf-8"),
        file_name="recommended_parts.csv",
        mime="text/csv",
        use_container_width=True,
    )


with tab_alt:
    require_solution()
    st.subheader("Alternatives (fast rules)")

    policies = st.session_state["heuristics"]
    rows = []
    for name, dpol in policies.items():
        mm = kpis(dpol)
        rows.append({
            "Strategy": name,
            "Total Net Savings": mm["total_net_savings"],
            "Fix-it-First": mm["fix_first"],
        })
    comp = pd.DataFrame(rows)

    st.dataframe(comp.assign(
        **{
            "Total Net Savings": comp["Total Net Savings"].map(money),
            "Fix-it-First": comp["Fix-it-First"].map(pct),
        }
    ), use_container_width=True)

    st.markdown("### Strategy comparison (Net Savings vs Fix-it-First)")

    comp2 = comp.sort_values("Total Net Savings", ascending=False).reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.6))

    x = range(len(comp2))
    ax1.bar(x, comp2["Total Net Savings"])
    ax1.set_ylabel("Total Net Savings ($)")
    ax1.set_xlabel("Strategy")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(comp2["Strategy"], rotation=25, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, comp2["Fix-it-First"], marker="o", color="red")
    ax2.set_ylabel("Fix-it-First (Demand Covered)")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.caption("Bars show annual net savings; the line shows first-visit coverage. Higher is better for both.")


with tab_risk:
    require_solution()
    st.subheader("Demand Uncertainty")

    inp = st.session_state["inputs"]
    df2 = st.session_state["df2"]

    st.markdown("""
Annual part usage can vary from year to year.  
This section stress-tests the plan by simulating many demand scenarios around the historical average,
then tracking how **total cost** and **fix-it-first rate** change.
""")

    st.info("Higher uncertainty usually increases variability (standard deviation) in cost and service coverage.")

    # choose which plan to stress test
    policies = st.session_state["heuristics"]
    plan_name = st.radio(
        "Plan to evaluate",
        ["Recommended (Optimal)"] + list(policies.keys()),
        index=0
    )
    portfolio = st.session_state["recommended"] if plan_name.startswith("Recommended") else policies[plan_name]

    # ---- 1) single risk level (Excel-like: line charts + histogram)
    st.markdown("### A) One risk level — simulation path + distribution")

    with st.spinner("Running simulations..."):
        sims = run_monte_carlo_fixed_portfolio_poisson(
            df2=df2,
            portfolio=portfolio,
            n_sims=int(inp["n_sims"]),
            seed=int(inp["seed"]),
            lam_scale=1.0  # base, uncertainty handled in sweep below; keep this for clean display
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Cost (mean)", money(sims["total_cost"].mean()))
    c2.metric("Fix-it-First (mean)", pct(sims["fix_first"].mean()))
    c3.metric("Net Savings (mean)", money(sims["total_net_savings"].mean()))

    # line: cost
    fig1, ax1 = plt.subplots(figsize=(7.2, 3.0))
    ax1.plot(sims["sim"], sims["total_cost"])
    ax1.set_title("Simulation Cost result")
    ax1.set_xlabel("Simulation run")
    ax1.set_ylabel("Total Cost ($)")
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)

    # line: fix-first
    fig2, ax2 = plt.subplots(figsize=(7.2, 3.0))
    ax2.plot(sims["sim"], sims["fix_first"])
    ax2.set_title("Simulation Fix-it-First percentage result")
    ax2.set_xlabel("Simulation run")
    ax2.set_ylabel("Fix-it-First")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    # histogram: cost distribution
    fig3, ax3 = plt.subplots(figsize=(7.2, 3.0))
    ax3.hist(sims["total_cost"], bins=12)
    ax3.set_title("Distribution of Simulated Total Costs")
    ax3.set_xlabel("Total Cost ($)")
    ax3.set_ylabel("Count")
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    # ---- 2) risk sweep 10%..50% (Excel-like: mean/std curves)
    st.markdown("### B) Risk level sweep (10% → 50%) — mean & standard deviation")

    with st.spinner("Running risk sweep..."):
        sweep = run_poisson_risk_sweep(
            df2=df2,
            portfolio=portfolio,
            risk_levels=(0.10, 0.20, 0.30, 0.40, 0.50),
            n_sims=int(inp["n_sims"]),
            seed=int(inp["seed"])
        )

    # cost mean
    fig4, ax4 = plt.subplots(figsize=(6.0, 2.8))
    ax4.plot(sweep["risk"], sweep["cost_mean"], marker="o")
    ax4.set_title("Cost Mean")
    ax4.set_xlabel("Risk level")
    ax4.set_ylabel("Cost mean ($)")
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)

    # cost std
    fig5, ax5 = plt.subplots(figsize=(6.0, 2.8))
    ax5.plot(sweep["risk"], sweep["cost_std"], marker="o")
    ax5.set_title("Cost Standard Deviation")
    ax5.set_xlabel("Risk level")
    ax5.set_ylabel("Cost std ($)")
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True)

    # fix mean
    fig6, ax6 = plt.subplots(figsize=(6.0, 2.8))
    ax6.plot(sweep["risk"], sweep["fix_mean"], marker="o")
    ax6.set_title("Fix-it-First percentage Mean")
    ax6.set_xlabel("Risk level")
    ax6.set_ylabel("Fix-it-First mean")
    fig6.tight_layout()
    st.pyplot(fig6, use_container_width=True)

    # fix std
    fig7, ax7 = plt.subplots(figsize=(6.0, 2.8))
    ax7.plot(sweep["risk"], sweep["fix_std"], marker="o")
    ax7.set_title("Fix-it-First percentage Standard Deviation")
    ax7.set_xlabel("Risk level")
    ax7.set_ylabel("Fix-it-First std")
    fig7.tight_layout()
    st.pyplot(fig7, use_container_width=True)

    with st.expander("Show sweep table"):
        st.dataframe(
            sweep.assign(
                cost_mean=sweep["cost_mean"].map(money),
                cost_std=sweep["cost_std"].map(lambda x: f"{x:,.2f}"),
                fix_mean=sweep["fix_mean"].map(pct),
                fix_std=sweep["fix_std"].map(lambda x: f"{x:.4f}"),
            ),
            use_container_width=True
        )