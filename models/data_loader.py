"""
Data loading and database seeding utilities.

Loads TSV files from the data/ directory and seeds the SQLite database
with customers and products.
"""
import ast
import os

import numpy as np
import pandas as pd


def load_data_from_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load customers and products from TSV files in data/."""
    data_dir = os.path.join(os.getcwd(), "data")

    customers_file = os.path.join(data_dir, "customers.tsv")
    products_file = os.path.join(data_dir, "products.tsv")

    for path in (customers_file, products_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")

    customers_df = pd.read_csv(customers_file, sep='\t')
    products_df = pd.read_csv(products_file, sep='\t')

    for name, df in [("customers.tsv", customers_df), ("products.tsv", products_df)]:
        if df.empty:
            raise ValueError(f"{name} is empty")

    return customers_df, products_df


def setup_real_data(customers_df: pd.DataFrame, products_df: pd.DataFrame) -> None:
    """Seed the database with customers and products from TSV data."""
    from models.models import Customer, Item

    segment_satisfaction = {'H': 0.8, 'R': 0.7, 'N': 0.6, 'L': 0.5, 'A': 0.4}

    for _, row in customers_df.iterrows():
        location = (np.random.uniform(-100, 100), np.random.uniform(-100, 100))
        income_factor = min(row['Income'] / 300000, 1.0)
        base_satisfaction = segment_satisfaction.get(row['Segment'], 0.6)
        satisfaction = min(base_satisfaction + income_factor * 0.2, 1.0)
        feature_vector = np.array(ast.literal_eval(row['FeatureVector'])).tolist()

        Customer.create(
            cid=row['CID'],
            name=row['Name'],
            age=row['Age'],
            gender=row['Gender'],
            location_x=location[0],
            location_y=location[1],
            job_type=row['JobType'],
            income=row['Income'],
            segment=row['Segment'],
            family_status=row['FamilyStatus'],
            feature_vector=feature_vector,
            satisfaction=satisfaction,
            visit_probability=0.1,
            session_probability=0.3,
            trigger_keywords=[],
        )

    for _, row in products_df.iterrows():
        cost = row['Cost'] if 'Cost' in row and pd.notna(row['Cost']) else row['Price'] * 0.5
        feature_vector = np.array(ast.literal_eval(row['FeatureVector'])).tolist()

        Item.create(
            pid=row['PID'],
            name=row['Name'],
            description=row['Description'],
            price=row['Price'],
            discounted=False,
            cost=cost,
            stock_level=np.random.randint(10, 101),
            feature_vector=feature_vector,
        )
