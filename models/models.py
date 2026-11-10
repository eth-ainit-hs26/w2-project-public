"""
Unified model definitions for the retail simulation database.

All Peewee ORM models live here.
"""
from peewee import BooleanField, CharField, DateField, FloatField, IntegerField, TextField

from models.database import BaseModel, JSONField


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------

class Customer(BaseModel):
    """Customer model with observable and hidden parameters."""

    cid = IntegerField(primary_key=True)
    name = CharField()
    age = IntegerField()
    gender = CharField()
    location_x = FloatField()
    location_y = FloatField()
    job_type = CharField()
    income = IntegerField()
    segment = CharField()
    family_status = CharField()
    satisfaction = FloatField(default=0.5)
    feature_vector = JSONField()              # pc ∈ R^D: preference vector
    preference_covariance = JSONField(null=True)  # Σc ∈ R^(D×D)
    visit_probability = FloatField(default=0.5)
    session_probability = FloatField(default=0.3)
    trigger_keywords = JSONField(default=list)

    class Meta:
        table_name = 'customers'

    @property
    def location(self) -> tuple[float, float]:
        return (self.location_x, self.location_y)

    @location.setter
    def location(self, value: tuple[float, float]):
        self.location_x, self.location_y = value


# ---------------------------------------------------------------------------
# Item
# ---------------------------------------------------------------------------

class Item(BaseModel):
    """Item model with observable and hidden feature parameters."""

    pid = IntegerField(primary_key=True)
    name = CharField()
    description = TextField()
    price = FloatField()
    discounted = BooleanField(default=False)
    cost = FloatField()
    stock_level = IntegerField(default=0)
    feature_vector = JSONField()              # fi ∈ R^D: item feature vector

    class Meta:
        table_name = 'products'


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """Transaction model (Figure 3 Database schema)."""

    tid = IntegerField(primary_key=True)
    cid = IntegerField()                      # Reference to Customer.cid
    date = DateField()
    order_data = JSONField()                  # List[(PID, quantity, price)]
    delivery_time_window_start = IntegerField()
    delivery_time_window_end = IntegerField()

    class Meta:
        table_name = 'transactions'

