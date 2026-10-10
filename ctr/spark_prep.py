"""PySpark preprocessing: raw Criteo TSV -> train/val/test Parquet.

Steps
  1. Read tab-separated raw logs (empty field -> null).
  2. Integer features: null -> 0, signed log1p to tame heavy tails, then standardise
     using mean/std computed on the TRAIN split only (no leakage into val/test).
  3. Categorical features: hash the string into [1, NUM_BUCKETS-1] with Spark's murmur3
     `hash`; null -> 0 ("missing" bucket). Fixed-size hashing keeps memory bounded for
     the very high-cardinality Criteo columns and needs no vocabulary pass.
  4. Split by row order (chronological in real Criteo): first 80% train, next 10% val,
     last 10% test. Cut points come from the monotonically increasing row id, so the
     split fractions are approximate (within ~0.1%), but the order is exact.

Usage:
    python -m ctr.spark_prep --input data/raw/train.txt --output data/processed
"""
import argparse
import json
import os

from pyspark.sql import SparkSession, functions as F, types as T

from ctr.config import (CAT_COLS, INT_COLS, LABEL, NUM_BUCKETS, RAW_COLUMNS,
                        TRAIN_FRAC, VAL_FRAC)


def build_spark(driver_memory: str | None = None, cores: str | int | None = None,
                shuffle_partitions: int | None = None) -> SparkSession:
    """Local-mode Spark. Override with env vars for big machines, e.g.
    SPARK_DRIVER_MEMORY=24g SPARK_CORES='*' SPARK_SHUFFLE_PARTITIONS=64 (defaults: 4g, 2, 8)."""
    driver_memory = driver_memory or os.environ.get("SPARK_DRIVER_MEMORY", "4g")
    cores = cores or os.environ.get("SPARK_CORES", "2")
    shuffle_partitions = shuffle_partitions or int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
    return (SparkSession.builder.master(f"local[{cores}]").appName("ctr-prep")
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .config("spark.ui.enabled", "false")
            # shuffle/spill files go here; on a cluster set SPARK_LOCAL_DIR=$SLURM_TMPDIR (node-local flash)
            .config("spark.local.dir", os.environ.get("SPARK_LOCAL_DIR", "/tmp"))
            .getOrCreate())


def read_raw(spark: SparkSession, path: str):
    schema = T.StructType([T.StructField(c, T.StringType(), True) for c in RAW_COLUMNS])
    return spark.read.csv(path, sep="\t", schema=schema, nullValue="", header=False)


def transform(df, num_buckets: int = NUM_BUCKETS):
    """Typed, cleaned columns (ints un-standardised yet) + a row-order id."""
    ints = [
        (F.signum(F.coalesce(F.col(c).cast("double"), F.lit(0.0)))
         * F.log1p(F.abs(F.coalesce(F.col(c).cast("double"), F.lit(0.0))))).alias(c)
        for c in INT_COLS
    ]
    cats = [
        F.when(F.col(c).isNull(), F.lit(0))
         .otherwise(F.lit(1) + F.pmod(F.hash(F.col(c)), F.lit(num_buckets - 1)))
         .cast("int").alias(c)
        for c in CAT_COLS
    ]
    return df.select(F.monotonically_increasing_id().alias("_rid"),
                     F.col(LABEL).cast("int").alias(LABEL), *ints, *cats)


def run(input_path: str, output_dir: str, num_buckets: int = NUM_BUCKETS) -> dict:
    spark = build_spark()
    try:
        df = transform(read_raw(spark, input_path), num_buckets).persist()
        n = df.count()

        # approximate row-order cut points from the (order-preserving) row id
        cut_train, cut_val = df.approxQuantile("_rid", [TRAIN_FRAC, TRAIN_FRAC + VAL_FRAC], 0.0005)
        train = df.filter(F.col("_rid") <= cut_train)
        val = df.filter((F.col("_rid") > cut_train) & (F.col("_rid") <= cut_val))
        test = df.filter(F.col("_rid") > cut_val)

        # train-only standardisation statistics
        agg = train.agg(*[F.mean(c).alias(f"m_{c}") for c in INT_COLS],
                        *[F.stddev(c).alias(f"s_{c}") for c in INT_COLS]).first()
        means = {c: float(agg[f"m_{c}"]) for c in INT_COLS}
        stds = {c: float(agg[f"s_{c}"]) or 1.0 for c in INT_COLS}

        def standardise(d):
            return d.select(LABEL, *[((F.col(c) - means[c]) / stds[c]).cast("float").alias(c) for c in INT_COLS],
                            *CAT_COLS)

        meta = {"num_buckets": num_buckets, "int_means": means, "int_stds": stds, "total_rows": n, "splits": {}}
        for name, part in (("train", train), ("val", val), ("test", test)):
            out = standardise(part)
            out.write.mode("overwrite").parquet(os.path.join(output_dir, name))
            cnt, clicks = part.count(), part.agg(F.sum(LABEL)).first()[0]
            meta["splits"][name] = {"rows": cnt, "ctr": clicks / cnt}

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta
    finally:
        spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/train.txt")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--num-buckets", type=int, default=NUM_BUCKETS)
    a = ap.parse_args()
    m = run(a.input, a.output, a.num_buckets)
    print(json.dumps({k: m[k] for k in ("total_rows", "splits")}, indent=2))
