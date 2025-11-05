import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
VOTES_CSV = (
    "/home/ahmedsalim/wrkspace/datasets/iaifi-hackathon-2025/data/target/gz2_hart16.csv"
)
MAP_CSV = "/home/ahmedsalim/wrkspace/datasets/iaifi-hackathon-2025/data/target/gz2_filename_mapping.csv"
IMG_DIR = "/home/ahmedsalim/wrkspace/datasets/iaifi-hackathon-2025/data/target/images_gz2/images"
AGN_CSV = "/home/ahmedsalim/wrkspace/datasets/iaifi-hackathon-2025/data/target/AGN_GZ2_Hart_DR7_final.csv"
OUT_FILE = "data/target/gz2_galaxy_labels.json"

# --- THRESHOLDS ---
THRESHOLDS = {
    "artifact": 0.2,
    "elliptical": 0.95,
    "nospiral": 0.9,
    "spiral": 0.95,
    "features": 0.9,
    "edgeon_ell": 0.1,
    "edgeon_spiral": 0.2,
    "odd": 0.7,
    "irregular": 0.7,
}


def load_data(votes_csv, map_csv, agn_csv=None):
    """
    Load and merge the Galaxy Zoo vote fractions and filename mapping.
    Returns a merged DataFrame with consistent 'objid' type.
    """
    votes_df = pd.read_csv(votes_csv, low_memory=False)
    map_df = pd.read_csv(map_csv, low_memory=False)

    if "dr7objid" in votes_df.columns:
        votes_df = votes_df.rename(columns={"dr7objid": "objid"})

    votes_df["objid"] = votes_df["objid"].astype(int)
    map_df["objid"] = map_df["objid"].astype(int)
    merged = votes_df.merge(map_df, on="objid", how="inner")

    if agn_csv is not None:
        agn_df = pd.read_csv(agn_csv, low_memory=False)
        assert ("dr7objid" in agn_df.columns) or (
            "OBJID" in agn_df.columns
        ), "AGN CSV must include 'dr7objid' or 'OBJID'"
        if "dr7objid" in agn_df.columns and "OBJID" in agn_df.columns:
            assert (
                agn_df["dr7objid"].astype(str).equals(agn_df["OBJID"].astype(str))
            ), "dr7objid and OBJID must be identical in AGN CSV"
        key = "dr7objid" if "dr7objid" in agn_df.columns else "OBJID"

        merged["objid_str"] = merged["objid"].astype(str)
        agn_df[key] = agn_df[key].astype(str)
        cols = [c for c in ["BPT_CLASS", "LOG_MSTELLAR", key] if c in agn_df.columns]
        merged = merged.merge(
            agn_df[cols], left_on="objid_str", right_on=key, how="left"
        )
        merged.drop(columns=["objid_str", key], inplace=True, errors="ignore")
    return merged


def classify_gz2(row):
    """
    Classify a galaxy into elliptical, spiral, or irregular
    using ultra-strict debiased vote fraction thresholds.
    Returns (label, metrics_dict).
    """
    m = {
        "artifact_prob": row["t01_smooth_or_features_a03_star_or_artifact_debiased"],
        "smooth_prob": row["t01_smooth_or_features_a01_smooth_debiased"],
        "features_prob": row["t01_smooth_or_features_a02_features_or_disk_debiased"],
        "edgeon_prob": row["t02_edgeon_a04_yes_debiased"],
        "spiral_prob": row["t04_spiral_a08_spiral_debiased"],
        "nospiral_prob": row["t04_spiral_a09_no_spiral_debiased"],
        "irregular_prob": row["t08_odd_feature_a22_irregular_debiased"],
        "merger_prob": row["t08_odd_feature_a24_merger_debiased"],
        "disturbed_prob": row["t08_odd_feature_a21_disturbed_debiased"],
        "odd_prob": row["t06_odd_a14_yes_debiased"],
    }

    if m["artifact_prob"] >= THRESHOLDS["artifact"]:
        return None, m

    if (
        m["smooth_prob"] >= THRESHOLDS["elliptical"]
        and m["edgeon_prob"] < THRESHOLDS["edgeon_ell"]
        and m["nospiral_prob"] >= THRESHOLDS["nospiral"]
    ):
        return "elliptical", m

    if (
        m["spiral_prob"] >= THRESHOLDS["spiral"]
        and m["features_prob"] >= THRESHOLDS["features"]
        and m["edgeon_prob"] < THRESHOLDS["edgeon_spiral"]
    ):
        return "spiral", m

    if (
        m["odd_prob"] >= THRESHOLDS["odd"]
        and max(m["irregular_prob"], m["merger_prob"], m["disturbed_prob"])
        >= THRESHOLDS["irregular"]
    ):
        return "irregular", m

    return None, m


def compute_mass_and_starformation(df):
    """
    Compute the stellar mass, star formation flag, and AGN presence
    for galaxies crossmatched with the Schawinski et al. (2010) AGN catalogue.

    How these values are derived:
    ---------------------------------
    • Stellar mass (mass):
        - Taken from the column LOG_MSTELLAR in the AGN catalogue (log10 of stellar mass in solar masses)
        - Converted to linear scale using: mass = 10 ** LOG_MSTELLAR
        - If LOG_MSTELLAR is missing, the mass is set to -1.0

    • Star formation flag (star_forming):
        - Based on the BPT_CLASS column (from the Baldwin–Phillips–Terlevich diagram classification)
        - star_forming = 1 if BPT_CLASS == 1 (pure star-forming galaxy)
        - star_forming = 0 otherwise

    • AGN presence flag (has_agn):
        - has_agn = 1 if BPT_CLASS is 3 (Seyfert) or 4 (LINER)
        - has_agn = 0 otherwise

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing BPT_CLASS and LOG_MSTELLAR columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns: 'mass', 'star_forming', and 'has_agn'
    """
    df = df.copy()

    # --- Stellar mass calculation ---
    df["mass"] = df["LOG_MSTELLAR"].apply(
        lambda x: float(np.power(10, x)) if pd.notna(x) else -1.0
    )

    # --- Star formation classification ---
    df["star_forming"] = df["BPT_CLASS"].apply(
        lambda x: 1 if pd.notna(x) and x == 1 else 0
    )

    # --- AGN classification ---
    df["has_agn"] = df["BPT_CLASS"].apply(
        lambda x: 1 if pd.notna(x) and x in [3, 4] else 0
    )

    return df


def find_image_path(asset_id):
    """
    find the image path for a given asset_id
    Returns the path if found, else None
    """
    patterns = [
        f"{asset_id}.jpg",
        # f"{int(asset_id)}.jpg"
    ]
    for pat in patterns:
        p = os.path.join(IMG_DIR, pat)
        if os.path.exists(p):
            return p
    return None


def top_n(df, n_per_class):
    """
    For each class, select the top N galaxies sorted by the strongest
    confidence metric relevant to that class.
    """
    best_rows = []
    for cls in ["elliptical", "spiral", "irregular"]:
        subset = df[df["classification"] == cls].copy()
        if cls == "elliptical":
            subset["sort_value"] = subset["metrics"].apply(lambda x: x["smooth_prob"])
        elif cls == "spiral":
            subset["sort_value"] = subset["metrics"].apply(lambda x: x["spiral_prob"])
        elif cls == "irregular":
            subset["sort_value"] = subset["metrics"].apply(
                lambda x: max(
                    x["irregular_prob"], x["merger_prob"], x["disturbed_prob"]
                )
            )
        subset = subset.sort_values(by="sort_value", ascending=False)
        best_rows.append(subset.head(n_per_class))
    return pd.concat(best_rows)


def classify_agn(row):
    """
    Derive AGN-related labels from merged columns:
    - star_forming: 1 if BPT_CLASS == 1 else 0
    - has_agn: 1 if BPT_CLASS in {3, 4} else 0
    - mass: 10**LOG_MSTELLAR (float) or -1.0 if missing
    Returns a dict with keys: star_forming, has_agn, mass
    """
    bpt = row.get("BPT_CLASS", None)
    log_m = row.get("LOG_MSTELLAR", None)

    star_forming = 1 if (pd.notna(bpt) and bpt == 1) else 0
    has_agn = 1 if (pd.notna(bpt) and bpt in [3, 4]) else 0
    mass = float(np.power(10, log_m)) if pd.notna(log_m) else -1.0

    return {"star_forming": star_forming, "has_agn": has_agn, "mass": mass}


def save_json(df, out_path):
    """
    Save the DataFrame to JSON with:
    - subhalo_id (image filename)
    - classification
    - metrics (vote fractions)
    - mass
    - star_forming
    - has_agn
    """
    data = [
        {
            "subhalo_id": row["subhalo_id"],
            "objid": int(row["objid"]),
            "classification": row["classification"],
            "mass": float(row["mass"]),
            "star_forming": int(row["star_forming"]),
            "has_agn": int(row["has_agn"]),
            "metrics": row["metrics"],
        }
        for _, row in df.iterrows()
    ]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} galaxies to {out_path}")


def save_csv(df, out_path):
    """
    Save the DataFrame to CSV with columns:
    - subhalo_id
    - objid
    - mass
    - star_forming
    - has_agn
    - classification
    """
    out_df = df[df["mass"] != -1.0].copy()
    out_df["star_forming"] = out_df["star_forming"].astype(bool)
    out_df["has_agn"] = out_df["has_agn"].astype(bool)
    out_df[
        ["subhalo_id", "objid", "mass", "star_forming", "has_agn", "classification"]
    ].to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} galaxies to {out_path}")


if __name__ == "__main__":
    df = load_data(VOTES_CSV, MAP_CSV, AGN_CSV)

    print("Classifying galaxies...")
    tqdm.pandas()
    df[["classification", "metrics"]] = df.progress_apply(
        lambda row: pd.Series(classify_gz2(row)), axis=1
    )

    # Derive AGN labels
    print("Deriving AGN labels (star_forming, has_agn, mass)...")
    agn_labels = df.progress_apply(classify_agn, axis=1)
    df["star_forming"] = agn_labels.apply(lambda d: d["star_forming"])
    df["has_agn"] = agn_labels.apply(lambda d: d["has_agn"])
    df["mass"] = agn_labels.apply(lambda d: d["mass"])

    print("Finding image paths...")
    df["image_path"] = df["asset_id"].progress_apply(find_image_path)

    valid_df = df[df["classification"].notnull() & df["image_path"].notnull()]
    valid_df["subhalo_id"] = valid_df["image_path"].apply(
        lambda x: os.path.basename(x) if x else None
    )
    print(valid_df["classification"].value_counts())

    # full masters
    save_json(valid_df, OUT_FILE.replace(".json", "_master.json"))
    save_csv(valid_df, OUT_FILE.replace(".json", "_master.csv"))

    # filtered top-N dataset
    filtered_df = valid_df[valid_df["mass"] != -1.0].copy()
    print("Filtered dataset shape:", filtered_df.shape)
    filtered_df = top_n(filtered_df, 2000)
    filtered_df = filtered_df.head(3000)
    save_json(filtered_df, OUT_FILE)
    save_csv(filtered_df, OUT_FILE.replace(".json", ".csv"))
