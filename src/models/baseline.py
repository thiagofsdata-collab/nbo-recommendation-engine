def compute_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    segment | product_name | acquisition_rate

    acquisition_rate = % of customers in that segment
                       who acquired that product in May
    """

    segment_size = (
        df.groupby("segment")["customer_id"]
        .nunique()
        .rename("n_customers")
    )

    acquisitions = (
        df[df["target"] == 1]
        .groupby(["segment", "product_name"])["customer_id"]
        .nunique()
        .rename("n_acquired")
    )

    result = acquisitions.reset_index().merge(
        segment_size.reset_index(),
        on="segment",
        how="left"
    )

    result["acquisition_rate"] = (
        result["n_acquired"] / result["n_customers"]
    )

    result = result[["segment", "product_name", "acquisition_rate"]]

    return result


def recommend_baseline(
    df: pd.DataFrame,
    popularity: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    For each customer, recommend top_k products
    they don't already own, ranked by segment popularity.

    Returns DataFrame with columns:
    customer_id | product_name | rank
    """

    already_owns = (
        df[df["total_products"] > 0]
        [["customer_id", "product_name", "has_current_account"]]
    )


    customer_profile = (
        df[["customer_id", "segment"]]
        .drop_duplicates()
    )

    owned = (
        df[df["total_products"] > 0]
        [["customer_id", "product_name"]]
        .drop_duplicates()
    )

    candidates = customer_profile.merge(
        popularity,
        on="segment",
        how="left"
    )

    candidates = candidates.merge(
        owned.assign(already_owns=1),
        on=["customer_id", "product_name"],
        how="left"
    )
    candidates = candidates[candidates["already_owns"].isna()].drop(columns="already_owns")

    candidates["rank"] = (
        candidates
        .groupby("customer_id")["acquisition_rate"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )

    candidates = candidates[candidates["rank"].notna()]
    candidates["rank"] = candidates["rank"].astype(int)

    result = (
        candidates[candidates["rank"] <= top_k]
        [["customer_id", "product_name", "rank"]]
        .sort_values(["customer_id", "rank"])
        .reset_index(drop=True)
    )

    return result


import pandas as pd

def evaluate_baseline(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 5
) -> float:
    """
    Calculates MAP@k.

    recommendations: customer_id | product_name | rank
    ground_truth: customer_id | product_name

    Returns MAP@k score (float).
    """

    recommendations = recommendations.sort_values(
        ["customer_id", "rank"]
    )

    gt = (
        ground_truth.groupby("customer_id")["product_name"]
        .apply(set)
        .to_dict()
    )

    ap_scores = []

    for customer_id, recs in recommendations.groupby("customer_id"):

        relevant_items = gt.get(customer_id, set())
        if not relevant_items:
            continue

        hits = 0
        precision_sum = 0.0

        for i, product in enumerate(recs["product_name"].head(k), start=1):

            if product in relevant_items:
                hits += 1
                precision = hits / i
                precision_sum += precision

        denom = min(len(relevant_items), k)
        if denom > 0:
            ap_scores.append(precision_sum / denom)

    if len(ap_scores) == 0:
        return 0.0

    return sum(ap_scores) / len(ap_scores)


if __name__ == "__main__":
    import os
    from sqlalchemy import create_engine
    from dotenv import load_dotenv

    load_dotenv()

    DB_URL = (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
        f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )

    engine = create_engine(DB_URL)

    print("Loading training dataset...")
    df = pd.read_sql("SELECT * FROM mart_training_dataset", engine)

    print(f"Dataset shape: {df.shape}")

    print("Computing popularity...")
    popularity = compute_popularity(df)

    print("Generating recommendations...")
    recommendations = recommend_baseline(df, popularity, top_k=5)

    print("Evaluating...")
    ground_truth = df[df["target"] == 1][["customer_id", "product_name"]]
    score = evaluate_baseline(recommendations, ground_truth, k=5)

    print(f"\nBaseline MAP@5: {score:.4f}")