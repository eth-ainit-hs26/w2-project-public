"""Plots for the data overview, training, κ prediction quality and the test simulation."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from utils.training import _prediction_vectors, print_training_summary


SEGMENT_NAMES = {"N": "Newcomers", "A": "At-risk", "H": "High-value", "L": "Low-value", "R": "Regular"}


def plot_data_overview(customer_registry, item_catalogue) -> None:
    """Products (price, margin) in the top row, customers (income, segment, age) in the bottom row."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    items = item_catalogue.get_all_items()
    customers = customer_registry.get_all_customers()
    prices = np.array([item.price for item in items], dtype=float)
    margins = np.array([item.price - item.cost for item in items], dtype=float)
    incomes = np.array([customer.income for customer in customers], dtype=float)
    ages = np.array([customer.age for customer in customers], dtype=float)
    segments = [customer.segment for customer in customers]

    income_ticks = [20_000, 50_000, 100_000, 200_000, 500_000]
    income_format = FuncFormatter(lambda v, _: f"${v / 1000:.0f}k")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    ax = axes[0, 0]
    ax.hist(prices, bins=15, color="#2E86AB", alpha=0.8, edgecolor="white")
    ax.set_title(f"Product Price ({len(items)} products)")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Number of products")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(margins, bins=15, color="#2E86AB", alpha=0.8, edgecolor="white")
    ax.set_title("Product Margin (price − cost)")
    ax.set_xlabel("Margin ($)")
    ax.set_ylabel("Number of products")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.scatter(prices, margins, alpha=0.6, s=30, color="#2E86AB")
    ax.set_title("Price vs. Margin")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Margin ($)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(incomes, bins=np.logspace(np.log10(incomes.min()), np.log10(incomes.max()), 13),
            color="#2E86AB", alpha=0.8, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xticks(income_ticks)
    ax.xaxis.set_major_formatter(income_format)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_title(f"Customer Income ({len(customers)} customers, log scale)")
    ax.set_xlabel("Annual income ($)")
    ax.set_ylabel("Number of customers")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    codes = [code for code in SEGMENT_NAMES if code in segments]
    by_segment = [incomes[[seg == code for seg in segments]] for code in codes]
    ax.boxplot(by_segment, patch_artist=True, widths=0.5, showfliers=False,
               boxprops=dict(facecolor="#2E86AB", alpha=0.35), medianprops=dict(color="black"))
    for pos, values in enumerate(by_segment, start=1):
        ax.scatter(np.full(len(values), pos), values, alpha=0.6, s=18, color="#2E86AB", zorder=3)
    ax.set_yscale("log")
    ax.set_yticks(income_ticks)
    ax.yaxis.set_major_formatter(income_format)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xticks(range(1, len(codes) + 1))
    ax.set_xticklabels([f"{SEGMENT_NAMES[code]}\n(n={len(v)})" for code, v in zip(codes, by_segment)])
    ax.set_title("Income by Customer Segment (log scale)")
    ax.set_ylabel("Annual income ($)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.hist(ages, bins=np.arange(15, 75, 5), color="#2E86AB", alpha=0.8, edgecolor="white")
    ax.set_title("Customer Age")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of customers")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def analyze_and_plot_kappa_prediction_quality(agent, customer_ids, item_ids, true_kappa_matrix):
    from sklearn.metrics import mean_squared_error

    predictions, truth = _prediction_vectors(agent, customer_ids, item_ids, true_kappa_matrix)
    correlation = float(np.corrcoef(truth, predictions)[0, 1]) if len(truth) > 1 else 0.0
    mse = float(mean_squared_error(truth, predictions))

    print(f"Correlation: r = {correlation:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Predicted range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"True range:      [{truth.min():.3f}, {truth.max():.3f}]")

    try:
        plot_kappa_prediction_quality(truth, predictions, correlation, mse)
    except Exception as e:
        print(f"(Plot skipped: {e})")
    return correlation, mse


def plot_learning_curves(history: Dict[str, Any], window: int = 10) -> None:
    """Training revenue, validation revenue and κ quality over the training iterations."""
    import matplotlib.pyplot as plt

    print_training_summary(history, window)

    iterations = np.arange(len(history["realized_revenue"]))
    realized = np.array(history["realized_revenue"])
    eval_iterations = history["eval_iterations"]
    validation = history["validation"]
    references = history["references"]
    global_oracle = references["Oracle"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    ax = axes[0]
    ax.plot(iterations, realized, color="#2E86AB", linewidth=0.6, alpha=0.5, label="Realized (dice)")
    if len(realized) >= window:
        moving = np.convolve(realized, np.ones(window) / window, mode="valid")
        ax.plot(iterations[window - 1:], moving, color="#2E86AB", linewidth=2.5,
                label=f"Realized, {window}-iteration mean")
    ax.plot(iterations, history["expected_revenue"], color="black", linestyle="--", linewidth=1.2,
            label="Expected (true κ)")
    ax.axhline(history["candidate_oracle"], color="red", linestyle=":", linewidth=1.5,
               label="Oracle within training items")
    ax.set_title("Training Revenue per Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Revenue ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pct = [100 * v["expected_revenue"] / global_oracle for v in validation]
    ax.plot(eval_iterations, pct, color="#2E86AB", marker="o", markersize=3, linewidth=2, label="CF agent.recommend()")
    for name, color in (("Highest Margin", "#F18F01"), ("Random", "#A23B72")):
        ax.axhline(100 * references[name] / global_oracle, color=color, linestyle="--", label=name)
    ax.set_title("Validation: Expected Revenue of Recommendations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("% of global oracle")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax, metric, title in ((axes[2], "kappa_r", "κ Correlation r"), (axes[3], "kappa_mse", "κ MSE")):
        ax.plot(eval_iterations, [v[f"train_{metric}"] for v in validation], color="gray",
                linestyle="--", label="Training pairs")
        ax.plot(eval_iterations, [v[f"val_{metric}"] for v in validation], color="navy",
                linewidth=2, label="Validation pairs")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_kappa_prediction_quality(true_values, predicted_values, correlation: float, mse: float) -> None:
    """Scatter of κ̂ vs κ, histogram of the errors κ̂ − κ, and errors against κ."""
    import matplotlib.pyplot as plt

    errors = np.asarray(predicted_values) - np.asarray(true_values)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.scatter(true_values, predicted_values, alpha=0.45, s=18, color="#2E86AB")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_title(f"Prediction Scatter (r={correlation:.3f}, MSE={mse:.4f})")
    ax.set_xlabel("True κ")
    ax.set_ylabel("Predicted κ̂")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(errors, bins=40, color="#2E86AB", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(errors.mean(), color="black", linewidth=1.5, label=f"Mean error {errors.mean():+.3f}")
    ax.set_title("Error Distribution (κ̂ − κ)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Number of pairs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(true_values, errors, alpha=0.45, s=18, color="#2E86AB")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Error vs. True κ")
    ax.set_xlabel("True κ")
    ax.set_ylabel("Error κ̂ − κ")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_comprehensive_performance_analysis(
    simulation_results: Dict[str, Dict[str, Any]],
    colors: Dict[str, str],
    oracle_daily_revenue: Optional[float] = None,
) -> None:
    """2x2 summary of the test simulation: cumulative revenue, totals, conversion and CF κ MSE.

    oracle_daily_revenue: expected revenue per day of a perfect oracle that knows the true κ
    (see oracle_expected_revenue); drawn as a dashed line in the cumulative revenue panel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    names = list(simulation_results.keys())
    n_days = len(simulation_results[names[0]]["daily_revenue"])
    days = np.arange(1, n_days + 1)
    agent_colors = [colors.get(n, "gray") for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    ax = axes[0]
    for name in names:
        cumrev = np.cumsum(simulation_results[name]["daily_revenue"])
        ax.plot(days, cumrev, color=colors.get(name, "gray"), label=name, linewidth=2)
    if oracle_daily_revenue is not None:
        ax.plot(days, oracle_daily_revenue * days, color="black", linestyle="--", linewidth=1.5,
                label="Oracle (expected, true κ)")
    ax.set_title("Cumulative Revenue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Total revenue (left axis, solid) and average conversion rate (right axis, hatched) per agent
    ax = axes[1]
    x = np.arange(len(names))
    width = 0.38
    tots = [simulation_results[n]["total_revenue"] for n in names]
    avgs = [simulation_results[n]["overall_conversion_rate"] for n in names]
    rev_bars = ax.bar(x - width / 2, tots, width, color=agent_colors, alpha=0.85)
    ax2 = ax.twinx()
    conv_bars = ax2.bar(x + width / 2, avgs, width, color=agent_colors, alpha=0.35,
                        hatch="//", edgecolor=agent_colors)
    for bar, v in zip(rev_bars, tots):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tots) * 0.01,
                f"${v / 1e6:.2f}M", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(conv_bars, avgs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avgs) * 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, max(tots) * 1.15)
    ax2.set_ylim(0, max(avgs) * 1.15)
    ax.set_ylabel("Total Revenue ($)")
    ax2.set_ylabel("Average Conversion Rate")
    ax.set_title("Total Revenue and Average Conversion Rate")
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.85, label="Total revenue (left)"),
                       Patch(facecolor="gray", alpha=0.35, hatch="//", edgecolor="gray",
                             label="Conversion rate (right)")],
              fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    for name in names:
        ax.plot(days, simulation_results[name]["daily_conversion_rate"],
                color=colors.get(name, "gray"), label=name, alpha=0.75)
    ax.set_title("Daily Conversion Rate")
    ax.set_xlabel("Day")
    ax.set_ylabel("Conversion Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    cf_name = "CF (Trained)"
    if cf_name in simulation_results:
        errs = simulation_results[cf_name]["daily_kappa_mse"]
        ax.plot(days[:len(errs)], errs, color=colors.get(cf_name, "blue"), linewidth=2, label="All pairs")
        shown_errs = simulation_results[cf_name].get("daily_shown_kappa_mse")
        if shown_errs:
            ax.plot(days[:len(shown_errs)], shown_errs, color=colors.get(cf_name, "blue"), linewidth=1,
                    linestyle="--", alpha=0.8, label="Shown products")
        ax.set_title("CF κ MSE (Test Simulation)")
        ax.set_xlabel("Day")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Test Performance: CF vs Baselines",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_exploration_runs(exploration_runs: Dict[float, Dict[str, Any]],
                          oracle_daily_revenue: Optional[float] = None) -> None:
    """Compare the exploration runs of 4.3: cumulative revenue, κ MSE on all pairs and recommend() after the test.

    ε = 0 is drawn in the CF colour of 4.2, larger ε in darker shades of orange/red; each ε has the
    same colour in all three panels.
    """
    import matplotlib.pyplot as plt

    epsilons = sorted(exploration_runs)
    positive = [eps for eps in epsilons if eps > 0]
    shades = np.linspace(0.35, 0.9, len(positive)) if len(positive) > 1 else [0.6] * len(positive)
    eps_colors = {0.0: "#2E86AB", **{eps: plt.cm.YlOrRd(s) for eps, s in zip(positive, shades)}}
    labels = {eps: f"ε = {eps:.2f}" + (" (reference)" if eps == 0 else "") for eps in epsilons}

    n_days = len(exploration_runs[epsilons[0]]["daily_revenue"])
    days = np.arange(1, n_days + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for eps in epsilons:
        ax.plot(days, np.cumsum(exploration_runs[eps]["daily_revenue"]), color=eps_colors[eps],
                linewidth=2, label=labels[eps])
    if oracle_daily_revenue is not None:
        ax.plot(days, oracle_daily_revenue * days, color="black", linestyle="--", linewidth=1.5,
                label="Oracle (expected, true κ)")
    ax.set_title("Cumulative Revenue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for eps in epsilons:
        ax.plot(days, exploration_runs[eps]["daily_kappa_mse"], color=eps_colors[eps], linewidth=2,
                label=labels[eps])
    ax.set_title("κ MSE on All Pairs")
    ax.set_xlabel("Day")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    x = np.arange(len(epsilons))
    pcts = [exploration_runs[eps]["final_recommend_pct_oracle"] for eps in epsilons]
    bars = ax.bar(x, pcts, color=[eps_colors[eps] for eps in epsilons], alpha=0.85)
    # Zoom the y-axis so that differences of a few percentage points are visible
    y_min = max(0.0, 5 * np.floor((min(pcts) - 5) / 5))
    y_max = max(pcts) + 3
    for bar, v in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * (y_max - y_min), f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε = {eps:.2f}" for eps in epsilons])
    ax.set_ylim(y_min, y_max)
    ax.set_title("recommend() After the Test")
    ax.set_ylabel("Expected revenue (% of oracle)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plt.show()
