# =============================================================
# Project : NBO Recommendation Engine
# Script  : src/ingestion/load_raw.py
# Purpose : Load Santander flat CSV into normalized PostgreSQL
#           tables using chunked reads to control memory.
# =============================================================

import os
import logging
from pathlib import Path
from io import StringIO
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

RAW_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/train_ver2.csv"))
CHUNK_SIZE = 50_000
SAMPLE_SEED = int(os.getenv("SAMPLE_SEED", 42))
SAMPLE_MONTHS = int(os.getenv("SAMPLE_MONTHS", 6))


# Column mapping

CUSTOMER_COLS = {
    "ncodpers" : "customer_id",
    "ind_empleado" : "employee_index",
    "pais_residencia": "country",
    "sexo" : "sex",
    "fecha_alta" : "first_contract_date",
    "indext" : "foreigner_index",
    "conyuemp" : "spouse_of_employee",
    "canal_entrada" : "acquisition_channel",
}


SNAPSHOT_COLS = {
    "ncodpers" : "customer_id",
    "fecha_dato" : "snapshot_date",
    "age" : "age",
    "antiguedad" : "seniority_months",
    "ind_nuevo" : "new_customer",
    "indrel" : "customer_type",
    "indrel_1mes" : "customer_type_month_start",
    "tiprel_1mes" : "relation_type_month_start",
    "ult_fec_cli_1t" : "last_date_as_primary",
    "indresi" : "residence_index",
    "ind_actividad_cliente" : "active_customer",
    "renta" : "gross_income",
    "segmento" : "segment",
    "cod_prov" : "province_code",
    "nomprov" : "province_name",
    "indfall" : "deceased",
}


PRODUCT_COLS = {
    "ind_ahor_fin_ult1" : "saving_account",
    "ind_aval_fin_ult1" : "guarantees",
    "ind_cco_fin_ult1" : "current_account",
    "ind_cder_fin_ult1" : "derivada_account",
    "ind_cno_fin_ult1" : "payroll_account",
    "ind_ctju_fin_ult1" : "junior_account",
    "ind_ctma_fin_ult1" : "mas_particular_account",
    "ind_ctop_fin_ult1" : "particular_account",
    "ind_ctpp_fin_ult1" : "particular_plus_account",
    "ind_deco_fin_ult1" : "short_term_deposit",
    "ind_deme_fin_ult1" : "medium_term_deposit",
    "ind_dela_fin_ult1" : "long_term_deposit",
    "ind_ecue_fin_ult1" : "e_account",
    "ind_fond_fin_ult1" : "funds",
    "ind_hip_fin_ult1" : "mortgage",
    "ind_plan_fin_ult1" : "pension_plan",
    "ind_pres_fin_ult1" : "loans",
    "ind_reca_fin_ult1" : "taxes",
    "ind_tjcr_fin_ult1" : "credit_card",
    "ind_valo_fin_ult1" : "securities",
    "ind_viv_fin_ult1" : "home_account",
    "ind_nomina_ult1" : "payroll",
    "ind_nom_pens_ult1" : "pensions",
    "ind_recibo_ult1" : "direct_debit",
}

def insert_df(df: pd.DataFrame, table: str, engine) -> None:
    """
    Insert DataFrame using PostgreSQL COPY FROM STDIN.
    Bypasses SQLAlchemy parameter limits entirely.
    10x faster than INSERT with method='multi'.
    """
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, na_rep="\\N")
        buffer.seek(0)
        cur.copy_expert(
            f"COPY {table} ({','.join(df.columns)}) FROM STDIN WITH CSV NULL '\\N'",
            buffer
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def bool_flag(series: pd.Series) -> pd.Series:
    """Convert S/N, 1/0, 1.0/0.0 to boolean. NaN → None."""
    mapped = series.map({
        "S": True, "N": False,
        1: True, 0: False,
        "1": True, "0": False,
        1.0: True, 0.0: False
    })
    return mapped.where(mapped.notna(), other=None)

def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    out = df[list(CUSTOMER_COLS.keys())].rename(columns=CUSTOMER_COLS).copy()

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip().replace("nan", None)

    out["foreigner_index"] = bool_flag(out["foreigner_index"])
    out["spouse_of_employee"] = bool_flag(out["spouse_of_employee"])
    out["first_contract_date"] = pd.to_datetime(out["first_contract_date"], errors="coerce")
    out["employee_index"] = out["employee_index"].str[:2]
    out["sex"] = out["sex"].str[:1]
    out["country"] = out["country"].str[:3]
    out["spouse_of_employee"] = out["spouse_of_employee"].where(
        out["spouse_of_employee"].notna(), other=None
    )

    out = out.drop_duplicates(subset=["customer_id"])
    return out


def clean_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    out = df[list(SNAPSHOT_COLS.keys())].rename(columns=SNAPSHOT_COLS).copy()

    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    out["last_date_as_primary"] = pd.to_datetime(out["last_date_as_primary"], errors="coerce")

    out["age"] = pd.to_numeric(out["age"].astype(str).str.strip(), errors="coerce").clip(0, 120).astype("Int16")
    out["seniority_months"] = pd.to_numeric(out["seniority_months"].astype(str).str.strip(), errors="coerce").clip(0).astype("Int32")

    out["gross_income"] = pd.to_numeric(out["gross_income"], errors="coerce")
    out["new_customer"] = bool_flag(out["new_customer"])
    out["residence_index"] = bool_flag(out["residence_index"])
    out["active_customer"] = bool_flag(out["active_customer"])
    out["deceased"] = out["deceased"].astype(str).str.strip().map({"S": True, "N": False}).where(
        lambda x: x.notna(), other=None
    )
    out["province_code"] = pd.to_numeric(out["province_code"], errors="coerce").astype("Int16")

    out["gross_income"] = out.groupby("segment")["gross_income"].transform(
        lambda x: x.fillna(x.median())
    )

    for col in ["customer_type", "customer_type_month_start"]:
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .replace("nan", None)
        )

    out["relation_type_month_start"] = (
        out["relation_type_month_start"]
        .astype(str).str.strip()
        .replace("nan", None)
    )

    out["segment"] = out["segment"].astype(str).str[:2].replace("na", None)

    return out


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    prod_df = df[["ncodpers", "fecha_dato"] + list(PRODUCT_COLS.keys())].copy()
    prod_df = prod_df.rename(columns={
        "ncodpers": "customer_id",
        "fecha_dato": "snapshot_date"
    })
    prod_df["snapshot_date"] = pd.to_datetime(prod_df["snapshot_date"], errors="coerce")

    prod_long = prod_df.melt(
        id_vars=["customer_id", "snapshot_date"],
        var_name="product_name",
        value_name="has_product"
    )
    prod_long["product_name"] = prod_long["product_name"].map(PRODUCT_COLS)

    prod_long["has_product"] = (
        pd.to_numeric(prod_long["has_product"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )
    return prod_long


# -----------------------------------------------------------
# Main ingestion
# -----------------------------------------------------------

def get_target_months(path: Path, n_months: int, seed: int) -> list:
    """Return the latest n_months from the available snapshot dates."""
    log.info("Scanning snapshot dates...")
    dates = pd.read_csv(path, usecols=["fecha_dato"])["fecha_dato"].unique()
    dates = sorted(dates)

    target = dates[-n_months:]
    log.info(f"Using {n_months} months: {target[0]} → {target[-1]}")
    return list(target)


def ingest(path: Path, engine, n_months: int, seed: int):
    target_months = get_target_months(path, n_months, seed)

    customers_seen = set()   
    chunk_n = 0

    log.info(f"Starting ingestion from {path} in chunks of {CHUNK_SIZE:,}...")

    for chunk in tqdm(pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False)):

        chunk = chunk[chunk["fecha_dato"].isin(target_months)]
        if chunk.empty:
            continue

        chunk_n += 1

        # customers (static, insert once per customer_id) 
        cust_df = clean_customers(chunk)
        new_customers = cust_df[~cust_df["customer_id"].isin(customers_seen)]
        if not new_customers.empty:
            with engine.begin() as conn:
                rows = new_customers.to_dict(orient="records")
                conn.execute(
                    text("""
                        INSERT INTO customers 
                            (customer_id, employee_index, country, sex,
                             first_contract_date, foreigner_index,
                             spouse_of_employee, acquisition_channel)
                        VALUES 
                            (:customer_id, :employee_index, :country, :sex,
                             :first_contract_date, :foreigner_index,
                             :spouse_of_employee, :acquisition_channel)
                        ON CONFLICT (customer_id) DO NOTHING
                    """),
                    rows
                )
            customers_seen.update(new_customers["customer_id"].tolist())

        # customer_snapshots
        snap_df = clean_snapshots(chunk)
        insert_df(snap_df, "customer_snapshots", engine)

        # customer_products
        prod_df = clean_products(chunk)
        insert_df(prod_df, "customer_products", engine)

        log.info(f"Chunk {chunk_n}: {len(chunk):,} rows → "
                 f"{len(new_customers):,} new customers, "
                 f"{len(snap_df):,} snapshots, "
                 f"{len(prod_df):,} product rows")

    log.info("Ingestion complete.")
    log.info(f"Total unique customers loaded: {len(customers_seen):,}")



if __name__ == "__main__":
    engine = create_engine(DB_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM product_catalog"))
        count = result.scalar()
        if count != 24:
            raise RuntimeError(f"product_catalog has {count} rows, expected 24. Run 01_create_tables.sql first.")
        log.info(f"Schema OK — product_catalog has {count} products.")

    ingest(RAW_PATH, engine, SAMPLE_MONTHS, SAMPLE_SEED)