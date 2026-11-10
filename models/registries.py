"""
Data-access registries and catalogues for all simulation models.
"""
from typing import List, Optional

from models.models import Customer, Item, Transaction


# ---------------------------------------------------------------------------
# CustomerRegistry
# ---------------------------------------------------------------------------

class CustomerRegistry:
    """Manages customers in the database."""

    def get_customer(self, customer_id: int) -> Optional[Customer]:
        try:
            return Customer.get(Customer.cid == customer_id)
        except Customer.DoesNotExist:
            return None

    def get_all_customers(self) -> List[Customer]:
        return list(Customer.select())


# ---------------------------------------------------------------------------
# ItemCatalogue
# ---------------------------------------------------------------------------

class ItemCatalogue:
    """Manages shop items in the database."""

    def get_item(self, item_id: int) -> Optional[Item]:
        try:
            return Item.get(Item.pid == item_id)
        except Item.DoesNotExist:
            return None

    def get_all_items(self) -> List[Item]:
        return list(Item.select())

    def get_available_items(self) -> List[Item]:
        return list(Item.select().where(Item.stock_level > 0))


# ---------------------------------------------------------------------------
# TransactionRegistry
# ---------------------------------------------------------------------------

class TransactionRegistry:
    """Manages transactions in the database."""

    def get_all_transactions(self) -> List[Transaction]:
        return list(Transaction.select())
