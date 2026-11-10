"""Test simulation of the agents and the final text analysis."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from utils.training import (
    _prediction_vectors,
    item_margins,
    oracle_expected_revenue,
    recommender_expected_revenue,
    select_training_items,
)


def clone_cf_agent(agent):
    """Copy of a CF agent with its own embeddings and biases (registries are shared)."""
    clone = copy.copy(agent)
    clone.U = {c: u.copy() for c, u in agent.U.items()}
    clone.V = {i: v.copy() for i, v in agent.V.items()}
    clone.b_c = dict(agent.b_c)
    clone.b_i = dict(agent.b_i)
    return clone


def detailed_agent_simulation(
    agent,
    agent_name: str,
    n_simulation_days: int,
    k: int,
    customer_registry,
    item_catalogue,
    customer_ids: List[int],
    item_ids: List[int],
    true_kappa_matrix: np.ndarray,
    epsilon: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the agent for n_simulation_days on the full catalogue; the agent keeps learning.

    With epsilon = 0 (default) the agent shows agent.recommend(c, k). With epsilon > 0 each
    of the k slots is filled with a random product with probability epsilon (ε-greedy, as in
    training), so the agent also gets feedback on products it would not recommend.
    """
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    customer_index = {customer_id: idx for idx, customer_id in enumerate(customer_ids)}
    margin = {item.pid: item.price - item.cost for item in item_catalogue.get_all_items()}
    rng = np.random.default_rng(seed)

    daily_revenue = []
    daily_purchases = []
    daily_attempts = []
    daily_conversion_rate = []
    daily_kappa_mse = []
    daily_shown_kappa_mse = []

    for _day in range(n_simulation_days):
        revenue = 0.0
        purchases = 0
        attempts = 0
        shown_pairs = []

        for customer_id in customer_ids:
            if epsilon > 0:
                shown_items = select_training_items(agent, customer_id, item_ids, margin, k, epsilon, rng)
            else:
                shown_items = agent.recommend(customer_id, k)
            purchased_items = []
            c_idx = customer_index[customer_id]
            for item_id in shown_items:
                i_idx = item_index[item_id]
                attempts += 1
                shown_pairs.append((customer_id, item_id, c_idx, i_idx))
                if np.random.random() < true_kappa_matrix[c_idx, i_idx]:
                    purchased_items.append(item_id)
                    purchases += 1
                    revenue += margin[item_id]

            agent.update_from_session(customer_id, shown_items, purchased_items)

        # κ MSE on all customer-item pairs, and on the pairs shown today
        preds, truth = _prediction_vectors(agent, customer_ids, item_ids, true_kappa_matrix)
        daily_kappa_mse.append(float(np.mean((preds - truth) ** 2)))
        daily_shown_kappa_mse.append(float(np.mean([
            (agent._kappa_hat(c, i) - true_kappa_matrix[c_idx, i_idx]) ** 2
            for c, i, c_idx, i_idx in shown_pairs
        ])) if shown_pairs else 0.0)
        daily_revenue.append(revenue)
        daily_purchases.append(purchases)
        daily_attempts.append(attempts)
        daily_conversion_rate.append(purchases / attempts if attempts else 0.0)

    total_revenue = float(np.sum(daily_revenue))
    total_purchases = int(np.sum(daily_purchases))
    total_attempts = int(np.sum(daily_attempts))
    return {
        "agent_name": agent_name,
        "daily_revenue": daily_revenue,
        "daily_purchases": daily_purchases,
        "daily_attempts": daily_attempts,
        "daily_conversion_rate": daily_conversion_rate,
        "daily_kappa_mse": daily_kappa_mse,
        "daily_shown_kappa_mse": daily_shown_kappa_mse,
        "total_revenue": total_revenue,
        "total_purchases": total_purchases,
        "total_attempts": total_attempts,
        "overall_conversion_rate": total_purchases / total_attempts if total_attempts else 0.0,
        "revenue_per_customer": total_revenue / max(1, len(customer_ids)),
    }


def run_exploration(
    epsilon: float,
    exploration_runs: Dict[float, Dict[str, Any]],
    trained_agent,
    n_simulation_days: int,
    k: int,
    customer_registry,
    item_catalogue,
    customer_ids: List[int],
    item_ids: List[int],
    true_kappa_matrix: np.ndarray,
    seed: int = 0,
) -> Dict[float, Dict[str, Any]]:
    """Run the test with ε-greedy exploration and store the result in exploration_runs (ε → result).

    Each run starts from a fresh copy of the trained agent with the same seed, so runs with
    different ε are comparable. An ε that is already in exploration_runs is not run again;
    ε = 0 is added once as the reference. Besides the simulation results, each run reports the
    expected revenue (true κ · margin) of agent.recommend() after the test as % of the oracle.
    """
    epsilon = round(float(epsilon), 4)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"EPSILON must be between 0 and 1, got {epsilon}")

    margins = item_margins(item_catalogue, item_ids)
    oracle = oracle_expected_revenue(true_kappa_matrix, margins, k)
    for eps in sorted({0.0, epsilon}):
        if eps in exploration_runs:
            print(f"ε = {eps:.2f}: already computed, using the stored result")
            continue
        print(f"ε = {eps:.2f}: running the {n_simulation_days}-day test ...")
        agent = clone_cf_agent(trained_agent)
        np.random.seed(seed)  # same purchase dice for every ε
        res = detailed_agent_simulation(agent, f"CF (ε={eps})", n_simulation_days, k, customer_registry,
                                        item_catalogue, customer_ids, item_ids, true_kappa_matrix,
                                        epsilon=eps, seed=seed)
        final_rev, _ = recommender_expected_revenue(agent, customer_ids, item_ids, true_kappa_matrix, margins, k)
        res["final_recommend_pct_oracle"] = 100 * final_rev / oracle
        exploration_runs[eps] = res
    return exploration_runs


def print_exploration_runs(exploration_runs: Dict[float, Dict[str, Any]]) -> None:
    """Table of all exploration runs, sorted by ε; "vs ε=0" is relative to the ε = 0 run."""
    base = exploration_runs[0.0]["total_revenue"]
    print(f"{'ε':>5} | {'test revenue':>13} | {'vs ε=0':>7} | {'κ MSE all pairs':>17} | "
          f"{'κ MSE shown':>11} | {'recommend() after test':>22}")
    print("-" * 92)
    for eps in sorted(exploration_runs):
        res = exploration_runs[eps]
        mse_all, mse_shown = res["daily_kappa_mse"], res["daily_shown_kappa_mse"]
        print(f"{eps:>5.2f} | ${res['total_revenue']:>12,.0f} | "
              f"{100 * (res['total_revenue'] - base) / base:>+6.1f}% | "
              f"{mse_all[0]:.3f} → {mse_all[-1]:.3f}     | "
              f"{np.mean(mse_shown):>11.3f} | "
              f"{res['final_recommend_pct_oracle']:>20.1f}% of oracle")


def print_comprehensive_analysis(
    simulation_results: Dict[str, Dict[str, Any]],
    correlation: float,
    mse: float,
    customer_registry,
) -> None:
    names = list(simulation_results.keys())
    by_rev = sorted(names, key=lambda n: simulation_results[n]["total_revenue"], reverse=True)
    by_conv = sorted(names, key=lambda n: simulation_results[n]["overall_conversion_rate"], reverse=True)

    print("COLLABORATIVE FILTERING VS BASELINES - FINAL ANALYSIS")
    print("=" * 60)
    print()
    print("PERFORMANCE RANKINGS:\n")
    print("By Total Revenue:")
    for rank, name in enumerate(by_rev, 1):
        rev = simulation_results[name]["total_revenue"]
        print(f"  {rank}. {name}: ${rev:,.0f}")

    print("\nBy Average Conversion Rate:")
    for rank, name in enumerate(by_conv, 1):
        conv = simulation_results[name]["overall_conversion_rate"]
        print(f"  {rank}. {name}: {conv:.3f}")

    print("\nKEY FINDINGS:\n")
    cf_name = "CF (Trained)"
    recommendations = []
    if cf_name in simulation_results:
        cf = simulation_results[cf_name]
        cf_rev = cf["total_revenue"]
        print("1. CF PERFORMANCE:")
        beaten, not_beaten = [], []
        for baseline in ("Random", "Highest Margin"):
            if baseline in simulation_results:
                b_rev = simulation_results[baseline]["total_revenue"]
                pct = (cf_rev - b_rev) / max(b_rev, 1) * 100
                if pct > 0:
                    beaten.append((baseline, pct))
                    print(f"   • CF outperforms {baseline} baseline by {pct:+.1f}%")
                else:
                    not_beaten.append(baseline)
                    print(f"   • CF does NOT beat {baseline} baseline ({pct:+.1f}%)")
        print(f"   • CF κ prediction quality: r={correlation:.3f}, MSE={mse:.4f}")
        if correlation >= 0.6:
            print("   • κ̂ tracks the true purchase probabilities well: CF has learned customer preferences")
        elif correlation >= 0.3:
            print("   • κ̂ tracks the true purchase probabilities only moderately")
        else:
            print("   • κ̂ barely tracks the true purchase probabilities: CF has not learned customer preferences")

        print("\n2. BASELINE COMPARISON:")
        for name in names:
            if name != cf_name:
                rev = simulation_results[name]["total_revenue"]
                conv = simulation_results[name]["overall_conversion_rate"]
                print(f"   • {name}: ${rev:,.0f} revenue, {conv:.3f} conversion")

        kappa_sim = cf.get("daily_kappa_mse", [])
        if kappa_sim:
            mse_change = (kappa_sim[-1] - kappa_sim[0]) / max(kappa_sim[0], 1e-10) * 100
            if abs(mse_change) < 0.05:
                trend = "unchanged"
            elif mse_change < 0:
                trend = f"improved by {-mse_change:.0f}%"
            else:
                trend = f"got worse by {mse_change:.0f}%"
            print(f"\n3. LEARNING EFFECTIVENESS:")
            print(f"   • CF κ MSE during simulation (all pairs, day 1 → day {len(kappa_sim)}): "
                  f"{kappa_sim[0]:.3f} → {kappa_sim[-1]:.3f} ({trend})")

            # Average daily revenue of the first and last 10 days
            daily = np.array(cf["daily_revenue"])
            window = min(10, len(daily))
            rev_start, rev_end = daily[:window].mean(), daily[-window:].mean()
            rev_change = (rev_end - rev_start) / max(rev_start, 1.0) * 100
            print(f"   • CF daily revenue, first vs last {window} days: "
                  f"${rev_start:,.0f} → ${rev_end:,.0f} ({rev_change:+.1f}%)")

        if not_beaten:
            recommendations.append(f"CF does not beat {', '.join(not_beaten)}: revisit the hyperparameters "
                                   "(TODO 6, tips in 5.2)")
        elif beaten:
            weakest, pct = min(beaten, key=lambda b: b[1])
            recommendations.append(f"Use CF as the primary recommender: it beats every baseline "
                                   f"(at least {pct:+.0f}% vs {weakest})")
        if correlation < 0.6:
            recommendations.append("Improve κ prediction quality (r < 0.6): tune d, lr and reg, "
                                   "check the learning curves in 3.3.1")

    print("\n4. BUSINESS INSIGHTS:")
    n_customers = len(customer_registry.get_all_customers())
    best = by_rev[0]
    rpc = simulation_results[best]["total_revenue"] / max(n_customers, 1)
    print(f"   • Best revenue per customer: ${rpc:,.0f}")

    cvs = {
        n: np.std(simulation_results[n]["daily_revenue"]) /
        max(np.mean(simulation_results[n]["daily_revenue"]), 1.0)
        for n in names
    }
    most_stable = min(cvs, key=cvs.get)
    print(f"   • Most stable revenue: {most_stable} (CV: {cvs[most_stable]:.3f})")
    print(f"   • Most efficient (purchases/attempts): {by_conv[0]} "
          f"({simulation_results[by_conv[0]]['overall_conversion_rate']:.3f})")

    print("\nRECOMMENDATIONS:")
    if not recommendations:
        recommendations.append(f"Use {best} as the primary recommender (highest total revenue)")
    for rec in recommendations:
        print(f"   • {rec}")
    print()
    print(f"CONCLUSION: {best} earns the most revenue in the 100-day test.")
