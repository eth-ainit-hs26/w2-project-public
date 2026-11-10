"""Helpers for the training loop and the validation of the CF agent."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _prediction_vectors(agent, customer_ids: List[int], item_ids: List[int], true_kappa_matrix: np.ndarray):
    predictions = []
    truth = []
    for c_idx, customer_id in enumerate(customer_ids):
        for i_idx, item_id in enumerate(item_ids):
            predictions.append(agent._kappa_hat(customer_id, item_id))
            truth.append(true_kappa_matrix[c_idx, i_idx])
    return np.array(predictions), np.array(truth)


def item_margins(item_catalogue, item_ids: List[int]) -> np.ndarray:
    """Margin (price - cost) per item, aligned with item_ids."""
    margin = {item.pid: item.price - item.cost for item in item_catalogue.get_all_items()}
    return np.array([margin[item_id] for item_id in item_ids], dtype=float)


def training_mask(customer_ids: List[int], item_ids: List[int], customer_training_items) -> np.ndarray:
    """Boolean (customers x items) matrix: True where the item is in the customer's training items.

    The complement (~mask) is the validation set: pairs the customer never sees during training.
    """
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    mask = np.zeros((len(customer_ids), len(item_ids)), dtype=bool)
    for c_idx, customer_id in enumerate(customer_ids):
        for item_id in customer_training_items[customer_id]:
            mask[c_idx, item_index[item_id]] = True
    return mask


def oracle_expected_revenue(true_kappa_matrix: np.ndarray, margins: np.ndarray, k: int, mask=None) -> float:
    """Expected revenue of showing every customer their best k items by true κ·margin.

    With a mask, only the items where mask is True are allowed (e.g. the training candidates).
    """
    expected = true_kappa_matrix * margins
    if mask is not None:
        expected = np.where(mask, expected, -np.inf)
    return float(np.sort(expected, axis=1)[:, -k:].sum())


def select_training_items(agent, customer_id: int, candidate_items: List[int], margin: Dict[int, float],
                          k: int, epsilon: float, rng: np.random.Generator) -> List[int]:
    """ε-greedy choice of the k items shown to a customer in one training session.

    Candidates are ranked by expected profit margin · κ̂. Each of the k slots gets the
    best remaining candidate, or, with probability epsilon, a random candidate that
    has not been chosen yet (exploration).
    """
    scores = {item_id: margin[item_id] * agent._kappa_hat(customer_id, item_id) for item_id in candidate_items}
    ranked = sorted(candidate_items, key=lambda item_id: scores[item_id], reverse=True)
    chosen: List[int] = []
    for _slot in range(min(k, len(ranked))):
        remaining = [item_id for item_id in ranked if item_id not in chosen]
        if rng.random() < epsilon:
            chosen.append(remaining[int(rng.integers(len(remaining)))])
        else:
            chosen.append(remaining[0])
    return chosen


def recommender_expected_revenue(recommender, customer_ids: List[int], item_ids: List[int],
                                 true_kappa_matrix: np.ndarray, margins: np.ndarray, k: int,
                                 train_mask=None):
    """Expected revenue (true κ·margin, no dice) of recommender.recommend(c, k) over all customers.

    Returns (expected_revenue, share of recommended items from the validation set);
    the share is None without a train_mask.
    """
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    expected_revenue = 0.0
    n_unseen = 0
    n_recommended = 0
    for c_idx, customer_id in enumerate(customer_ids):
        for item_id in recommender.recommend(customer_id, k):
            i_idx = item_index[item_id]
            expected_revenue += true_kappa_matrix[c_idx, i_idx] * margins[i_idx]
            n_recommended += 1
            if train_mask is not None and not train_mask[c_idx, i_idx]:
                n_unseen += 1
    share_unseen = n_unseen / max(n_recommended, 1) if train_mask is not None else None
    return float(expected_revenue), share_unseen


def validation_reference_revenues(item_catalogue, customer_ids: List[int], item_ids: List[int],
                                  true_kappa_matrix: np.ndarray, margins: np.ndarray, k: int) -> Dict[str, float]:
    """Expected revenue of the global oracle and the two (non-learning) baselines on the validation metric."""
    from agents.highest_margin_recommender import HighestMarginRecommenderAgent

    highest_margin = HighestMarginRecommenderAgent(None, item_catalogue, None)
    highest_margin_revenue, _ = recommender_expected_revenue(
        highest_margin, customer_ids, item_ids, true_kappa_matrix, margins, k)
    return {
        "Oracle": oracle_expected_revenue(true_kappa_matrix, margins, k),
        "Highest Margin": highest_margin_revenue,
        # exact expectation of k uniformly random items per customer
        "Random": float(k * (true_kappa_matrix * margins).mean(axis=1).sum()),
    }


def evaluate_validation(agent, customer_ids: List[int], item_ids: List[int], true_kappa_matrix: np.ndarray,
                        margins: np.ndarray, train_mask: np.ndarray, k: int) -> Dict[str, float]:
    """Validation during training. Never updates the agent and never samples purchases.

    - recommendation validation: expected revenue of agent.recommend(c, k) over the full catalogue
      and the share of recommended items the customer never saw in training
    - κ validation: MSE and correlation of κ̂ vs κ on validation pairs, training pairs reported separately
    """
    expected_revenue, share_unseen = recommender_expected_revenue(
        agent, customer_ids, item_ids, true_kappa_matrix, margins, k, train_mask)
    predictions, _ = _prediction_vectors(agent, customer_ids, item_ids, true_kappa_matrix)
    predictions = predictions.reshape(true_kappa_matrix.shape)
    metrics = {"expected_revenue": expected_revenue, "share_unseen": share_unseen}
    for name, mask in (("train", train_mask), ("val", ~train_mask)):
        pred, truth = predictions[mask], true_kappa_matrix[mask]
        metrics[f"{name}_kappa_mse"] = float(np.mean((pred - truth) ** 2))
        metrics[f"{name}_kappa_r"] = float(np.corrcoef(pred, truth)[0, 1])
    return metrics


def print_training_summary(history: Dict[str, Any], window: int = 10) -> None:
    """Start → end summary of training revenue, validation revenue and κ quality."""
    expected = np.array(history["expected_revenue"])
    candidate_oracle = history["candidate_oracle"]
    validation = history["validation"]
    references = history["references"]
    global_oracle = references["Oracle"]
    first, last = validation[0], validation[-1]

    print("TRAINING (agent chooses K of each customer's training items)")
    print(f"   • Expected revenue: {100 * expected[0] / candidate_oracle:.1f}% → "
          f"{100 * expected[-window:].mean() / candidate_oracle:.1f}% of the candidate oracle "
          f"(first iteration → mean of last {window})")
    print("\nVALIDATION (agent.recommend() on the full catalogue, true κ, no dice)")
    print(f"   • CF expected revenue: {100 * first['expected_revenue'] / global_oracle:.1f}% → "
          f"{100 * last['expected_revenue'] / global_oracle:.1f}% of the global oracle")
    for name in ("Highest Margin", "Random"):
        print(f"   • {name}: {100 * references[name] / global_oracle:.1f}% of the global oracle")
    print(f"   • Recommended items never shown to the customer in training: "
          f"{100 * first['share_unseen']:.0f}% → {100 * last['share_unseen']:.0f}%")
    print(f"   • κ on validation pairs: MSE {first['val_kappa_mse']:.4f} → {last['val_kappa_mse']:.4f}, "
          f"r {first['val_kappa_r']:.3f} → {last['val_kappa_r']:.3f}")
    print(f"   • κ on training pairs:   MSE {first['train_kappa_mse']:.4f} → {last['train_kappa_mse']:.4f}, "
          f"r {first['train_kappa_r']:.3f} → {last['train_kappa_r']:.3f}")
