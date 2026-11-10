"""Environment setup, data loading, ground-truth purchase probabilities and the TODO check."""
from __future__ import annotations

import os
import sys

import numpy as np


def _project1_dir() -> str:
    """Return the absolute path to the Project1 directory (parent of this utils/ folder)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def setup_environment(verbose: bool = True) -> None:
    """Configure imports and working directory for the part 2 notebooks."""
    project1_dir = _project1_dir()
    if project1_dir not in sys.path:
        sys.path.insert(0, project1_dir)
    os.chdir(project1_dir)
    np.random.seed(42)
    if verbose:
        print("Environment configured for CF analysis")


def initialize_system_components(
    n_iterations: int = 300,
    n_simulation_days: int = 100,
    k: int = 5,
    items_per_customer: int = 15,
):
    """Reset and seed the database with customer/item data, return registries for part 2."""
    setup_environment(verbose=False)  # re-seed silently; the notebook already printed the message

    from models.data_loader import load_data_from_files, setup_real_data
    from models.database import database, initialize_database
    from models.models import Customer, Item, Transaction
    from models.registries import CustomerRegistry, ItemCatalogue, TransactionRegistry

    if database.is_closed():
        database.connect()
    database.drop_tables([Transaction, Customer, Item], safe=True)
    initialize_database()

    customers_df, products_df = load_data_from_files()
    setup_real_data(customers_df, products_df)

    customer_registry = CustomerRegistry()
    item_catalogue = ItemCatalogue()
    transaction_registry = TransactionRegistry()
    hyperparams = {
        "N_ITERATIONS": n_iterations,
        "N_SIMULATION_DAYS": n_simulation_days,
        "K": k,
        "ITEMS_PER_CUSTOMER": items_per_customer,
    }

    n_customers = len(customer_registry.get_all_customers())
    n_items = len(item_catalogue.get_all_items())
    n_transactions = len(transaction_registry.get_all_transactions())

    print("System components initialized:")
    print(f"   • Customers: {n_customers}")
    print(f"   • Items: {n_items}")
    print(f"   • Transactions: {n_transactions}")
    print(f"   • Training items per customer: {items_per_customer}")
    print(f"   • Recommendation size: {k}")
    return customer_registry, item_catalogue, transaction_registry, hyperparams


KAPPA_SHARPNESS = 6.0
KAPPA_THRESHOLD = 0.5


def _true_purchase_probability(customer_vector: np.ndarray, item_vector: np.ndarray) -> float:
    """Ground-truth purchase probability κ(c, i) = sigmoid(6 * cos(c, i) - 3).

    cos is the cosine similarity of the 6-D vectors (5 taste dimensions plus
    the weighted budget/price dimension, see data/add_financial_feature.py).
    The sigmoid is centred at cos = 0.5, so only clearly matching
    customer-item pairs are likely purchases.
    """
    denom = np.linalg.norm(customer_vector) * np.linalg.norm(item_vector)
    if denom == 0:
        return 0.0
    cosine = float(np.dot(customer_vector, item_vector) / denom)
    return float(1.0 / (1.0 + np.exp(-KAPPA_SHARPNESS * (cosine - KAPPA_THRESHOLD))))


def prepare_ground_truth_data(customer_registry, item_catalogue):
    """Return customer IDs, item IDs and the true κ matrix (customers × items)."""
    customers = customer_registry.get_all_customers()
    items = item_catalogue.get_all_items()
    customer_ids = [customer.cid for customer in customers]
    item_ids = [item.pid for item in items]
    customer_preferences = np.array([customer.feature_vector for customer in customers], dtype=float)
    item_features = np.array([item.feature_vector for item in items], dtype=float)

    true_kappa_matrix = np.zeros((len(customers), len(items)), dtype=float)
    for c_idx, customer_vector in enumerate(customer_preferences):
        for i_idx, item_vector in enumerate(item_features):
            true_kappa_matrix[c_idx, i_idx] = _true_purchase_probability(customer_vector, item_vector)

    return customer_ids, item_ids, true_kappa_matrix


def check_todo_implementations(customer_registry, item_catalogue, transaction_registry) -> bool:
    from agents.recommender_agent import RecommenderAgent

    print("CHECKING YOUR TODO IMPLEMENTATIONS")
    print("=" * 50)
    print()

    try:
        agent = RecommenderAgent(
            customer_registry, item_catalogue, transaction_registry,
            d=3, lr=0.01, reg=1e-4, w_purchase=1.0, w_neg=0.5,
        )
    except Exception as e:
        print(f"❌ Failed to create RecommenderAgent: {e}")
        return False

    customers = customer_registry.get_all_customers()
    items = item_catalogue.get_all_items()
    if not customers or not items:
        print("❌ No customers or items in registry — run initialize_system_components first.")
        return False

    test_cid = customers[0].cid
    test_iids = [it.pid for it in items[:3]]

    print(f"Using test customer: {test_cid}, test item: {test_iids[0]}")
    print()

    all_ok = True

    print("1. Testing _kappa_hat() function...")
    try:
        kappa = agent._kappa_hat(test_cid, test_iids[0])
        if 0.0 <= kappa <= 1.0:
            print(f"   ✅ κ prediction: {kappa:.4f} (valid range [0,1])")
        else:
            print(f"   ❌ κ prediction: {kappa:.4f} (INVALID — must be in [0,1])")
            all_ok = False
    except NotImplementedError:
        print("   ❌ _kappa_hat() not yet implemented")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error in _kappa_hat(): {e}")
        all_ok = False

    print("\n2. Testing recommend() function...")
    try:
        recs = agent.recommend(test_cid, k=3)
        all_item_ids = {it.pid for it in items}
        if isinstance(recs, list) and len(recs) > 0:
            print(f"   ✅ Recommendations: {recs} (valid list of {len(recs)} items)")
            if all(r in all_item_ids for r in recs):
                print("   ✅ All recommended items exist in catalogue")
            else:
                print("   ❌ Some recommended items not found in catalogue")
                all_ok = False
        else:
            print(f"   ❌ Invalid recommendations: {recs}")
            all_ok = False
    except NotImplementedError:
        print("   ❌ recommend() not yet implemented")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error in recommend(): {e}")
        all_ok = False

    print("\n3. Testing update_from_session() function...")
    try:
        agent.update_from_session(test_cid, test_iids, [test_iids[0]])
        print("   ✅ update_from_session() executed without errors")
        print(f"   ✅ Processed {len(test_iids)} shown items, 1 purchases")
    except NotImplementedError:
        print("   ❌ update_from_session() not yet implemented")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error in update_from_session(): {e}")
        all_ok = False

    print("\n4. Testing _update_embeddings() function...")
    try:
        agent._update_embeddings(test_cid, test_iids[0], target=1.0, weight=1.0)
        print("   ✅ Embeddings updated successfully")
    except NotImplementedError:
        print("   ❌ _update_embeddings() not yet implemented")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error in _update_embeddings(): {e}")
        all_ok = False

    print("\n5. Testing function integration...")
    try:
        k_before = agent._kappa_hat(test_cid, test_iids[0])
        agent._update_embeddings(test_cid, test_iids[0], target=0.0, weight=1.0)
        k_after = agent._kappa_hat(test_cid, test_iids[0])
        if abs(k_before - k_after) > 1e-8:
            print(f"   ✅ κ prediction changed from {k_before:.4f} to {k_after:.4f} (learning working)")
        else:
            print("   ⚠️  κ prediction unchanged — gradient updates may not be working")
    except Exception as e:
        print(f"   ❌ Error in integration test: {e}")
        all_ok = False

    print()
    print("=" * 50)
    if all_ok:
        print("CHECK COMPLETE!")
        print("✅ Your implementations appear to be working correctly.")
        print("You can now proceed to run the full simulation.")
    else:
        print("❌ Some checks failed. Please review your implementations.")
    print()
    return all_ok
