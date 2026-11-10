"""
Append the financial feature (6th dimension) to the feature vectors in
customers.tsv and products.tsv.

The first five dimensions are the original Gemini-generated features
(style, functionality, durability, sustainability, innovation). The sixth
dimension models money:

- customers: budget b_c = 2 * (ln(Income_c) - ln(Income_min)) / (ln(Income_max) - ln(Income_min)) - 1,
  i.e. log income scaled to [-1, 1] (richest customer = +1)
- products:  price p_i = -(2 / (1 + exp(0.02 * (Price_i - 180))) - 1),
  a sigmoid centred at $180 scaled to [-1, 1] (expensive = +1)

Both are multiplied by the weight W_FINANCIAL = 2 before being appended, so
the financial match counts more than a single taste dimension. Wealthy
customers lean towards expensive products, budget customers lean away from
them. Prices, costs and the first five dimensions stay unchanged.

The script is idempotent: it always keeps only the first five entries of each
vector and recomputes the sixth, so running it twice gives the same files.

Usage (from the repository root or from data/):
    python data/add_financial_feature.py
"""
import ast
import csv
import os

import numpy as np

N_BASE_FEATURES = 5
W_FINANCIAL = 2.0
PRICE_SCALE = 0.02
PRICE_CENTER = 180.0

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def budget_feature(income: np.ndarray) -> np.ndarray:
    """Log income scaled to [-1, 1]."""
    log_income = np.log(income)
    return 2 * (log_income - log_income.min()) / (log_income.max() - log_income.min()) - 1


def price_feature(price: np.ndarray) -> np.ndarray:
    """Sigmoid of the price scaled to [-1, 1]; expensive items get values close to +1."""
    return -(2 / (1 + np.exp(PRICE_SCALE * (price - PRICE_CENTER))) - 1)


def _format_vector(values) -> str:
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


def _rewrite_feature_vectors(filename: str, value_column: str, feature_fn) -> None:
    """Replace the FeatureVector column with [5 original values, W * feature].

    Rows are rewritten line by line so that the rest of the file (quoting,
    column order, missing trailing newline) stays exactly as it was.
    """
    path = os.path.join(DATA_DIR, filename)
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    trailing_newline = text.endswith("\n")
    lines = text.rstrip("\n").split("\n")
    header = lines[0].split("\t")
    vector_idx = header.index("FeatureVector")
    value_idx = header.index(value_column)
    assert vector_idx == len(header) - 1, "FeatureVector is expected to be the last column"

    rows = list(csv.reader(lines[1:], delimiter="\t"))
    base_vectors = []
    for row in rows:
        vector = ast.literal_eval(row[vector_idx])
        assert len(vector) in (N_BASE_FEATURES, N_BASE_FEATURES + 1), f"unexpected vector length {len(vector)}"
        base_vectors.append(vector[:N_BASE_FEATURES])
    values = np.array([float(row[value_idx]) for row in rows])
    financial = W_FINANCIAL * feature_fn(values)

    new_lines = [lines[0]]
    for line, base, extra in zip(lines[1:], base_vectors, financial):
        prefix, old_field = line.rsplit("\t", 1)
        new_field = _format_vector(base + [round(float(extra), 4) + 0.0])  # +0.0 turns -0.0 into 0.0
        if old_field.startswith('"'):
            new_field = f'"{new_field}"'
        new_lines.append(f"{prefix}\t{new_field}")

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(new_lines) + ("\n" if trailing_newline else ""))
    print(f"{filename}: {len(rows)} rows, financial feature range "
          f"[{financial.min():.3f}, {financial.max():.3f}]")


def main() -> None:
    _rewrite_feature_vectors("customers.tsv", "Income", budget_feature)
    _rewrite_feature_vectors("products.tsv", "Price", price_feature)


if __name__ == "__main__":
    main()
