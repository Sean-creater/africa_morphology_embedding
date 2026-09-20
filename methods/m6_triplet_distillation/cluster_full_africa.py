"""
Multi-resolution KMeans clustering on M6 embeddings (k=1..30).

Loads all per-city .npz files produced by extract_full_africa.py, runs
MiniBatchKMeans for each k, and saves a single CSV with columns:
  grid_id, country, city, k1, k2, ..., k30

KMeans is fit on L2-normalised embeddings with cosine-equivalent geometry
(since embeddings are unit vectors, Euclidean distance ~ cosine distance).

EMB_ROOT defaults to the same directory extract_full_africa.py writes to;
override with the MORPHOLOGY_EMBEDDINGS_ROOT environment variable.
"""
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import CLEANED_CSV_ROOT

EMB_ROOT   = Path(os.environ.get(
    "MORPHOLOGY_EMBEDDINGS_ROOT",
    str(CLEANED_CSV_ROOT.parent / "data_embeddings_full_africa"),
))
OUT_CSV    = EMB_ROOT / "cluster_labels_k1_k30.csv"
LOG_PATH   = EMB_ROOT / "cluster.log"
K_MIN, K_MAX = 1, 30
BATCH_SIZE = 100_000
RANDOM_SEED = 42


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )
    return logging.getLogger()


def load_all_embeddings(emb_root, log):
    log.info(f"Loading embeddings from {emb_root} ...")
    t0 = time.time()

    emb_chunks, grid_ids, countries, cities = [], [], [], []
    npz_paths = sorted(emb_root.rglob("*.npz"))
    log.info(f"  Found {len(npz_paths):,} .npz files")

    for path in npz_paths:
        country = path.parent.name
        city    = path.stem
        d = np.load(path, allow_pickle=True)
        embs = d["embeddings"].astype(np.float32)
        gids = d["grid_ids"].astype(str)
        n = len(gids)
        emb_chunks.append(embs)
        grid_ids.extend(gids)
        countries.extend([country] * n)
        cities.extend([city] * n)

    embeddings = np.vstack(emb_chunks)
    log.info(f"  Loaded {len(grid_ids):,} cells  shape={embeddings.shape}  "
             f"[{time.time()-t0:.1f}s]")
    return embeddings, grid_ids, countries, cities


def run_clustering(embeddings, log):
    n = len(embeddings)
    all_labels = {}

    for k in range(K_MIN, K_MAX + 1):
        t0 = time.time()

        if k == 1:
            labels = np.zeros(n, dtype=np.int32)
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=BATCH_SIZE,
                random_state=RANDOM_SEED,
                n_init=3,
                max_iter=100,
                verbose=0,
            )
            labels = kmeans.fit_predict(embeddings).astype(np.int32)

        all_labels[f"k{k}"] = labels
        log.info(f"  k={k:2d}  elapsed={time.time()-t0:.1f}s")

    return all_labels


def main():
    log = setup_logging()
    log.info("=" * 60)
    log.info("Multi-resolution KMeans clustering  k=1..30")
    log.info("=" * 60)

    embeddings, grid_ids, countries, cities = load_all_embeddings(EMB_ROOT, log)

    log.info(f"\nRunning KMeans for k={K_MIN}..{K_MAX} ...")
    t_cluster = time.time()
    all_labels = run_clustering(embeddings, log)
    log.info(f"Clustering done in {(time.time()-t_cluster)/60:.1f} min")

    log.info("\nBuilding output table ...")
    df = pd.DataFrame({
        "grid_id": grid_ids,
        "country": countries,
        "city":    cities,
    })
    for col, labels in all_labels.items():
        df[col] = labels

    df.to_csv(OUT_CSV, index=False)
    log.info(f"Saved: {OUT_CSV}  ({len(df):,} rows x {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
