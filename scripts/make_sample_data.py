"""Generate a synthetic churn dataset for local training and tests.

This is NOT real customer data. It is randomly generated with a handful of
rules baked in (short tenure, month-to-month contracts, and lots of support
tickets push churn probability up) so the pipeline has something with a real
signal to train against. Numbers reported from `make train` / `make eval`
come from running the code on this synthetic data, not from a real telecom
dataset.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTRACT_TYPES = ["monthly", "one_year", "two_year"]
PAYMENT_METHODS = ["credit_card", "bank_transfer", "auto_pay", "mailed_check"]


def generate_churn_data(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic customer table with a churn label.

    Churn probability is driven by a logistic function of tenure, contract
    type, and support ticket volume, plus random noise, so the label
    correlates with the features without being a deterministic function of
    them.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, size=n_rows)
    monthly_charges = np.round(rng.uniform(20, 120, size=n_rows), 2)
    num_support_tickets = rng.poisson(1.5, size=n_rows)
    avg_monthly_usage = rng.integers(10, 500, size=n_rows)
    contract_type = rng.choice(CONTRACT_TYPES, size=n_rows, p=[0.55, 0.3, 0.15])
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows)

    # total_charges roughly tracks tenure * monthly_charges with noise, the
    # way an accumulated billing total would in a real dataset.
    total_charges = np.round(tenure * monthly_charges * rng.uniform(0.9, 1.1, size=n_rows), 2)

    contract_risk = np.where(contract_type == "monthly", 1.4, np.where(contract_type == "one_year", 0.0, -1.0))
    logit = -1.2 - 0.04 * tenure + 0.25 * num_support_tickets + contract_risk + 0.01 * (monthly_charges - 60)
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob)

    customer_id = [f"CUST-{i:06d}" for i in range(n_rows)]

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "avg_monthly_usage": avg_monthly_usage,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "churn": churn,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/churn_data.csv")
    parser.add_argument("--n-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = generate_churn_data(n_rows=args.n_rows, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "Wrote %d synthetic rows to %s (churn rate: %.1f%%)",
        len(df),
        output_path,
        100 * df["churn"].mean(),
    )


if __name__ == "__main__":
    main()
