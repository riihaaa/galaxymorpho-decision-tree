"""
Step 0: Build a clean, labeled GZ2 dataset ready for ML.

Inputs you need locally (not in this sandbox):
  - gz2_hart16_csv.gz               (you already have this)
  - gz2_filename_mapping.csv        (download from the SAME Zenodo record, 3565489
                                      -> "gz2_filename_mapping.csv", ~13MB)
  - images_gz2.zip                  (the 3.4GB image set -> unzip locally, e.g. to ./images_gz2/)

WHY the mapping file matters:
  The catalog (gz2_hart16.csv) indexes galaxies by `dr7objid` (SDSS DR7 object ID).
  The images inside images_gz2.zip are named by a DIFFERENT id (an internal Galaxy
  Zoo "asset id", e.g. "100018.jpg"), NOT by dr7objid. gz2_filename_mapping.csv has
  3 columns: objid (=dr7objid), sample, asset_id -- this is what lets you go from a
  catalog row to the actual .jpg file on disk. Skipping this step is the #1 reason
  people get "half my galaxies have no image" bugs.

Run this once, it produces `gz2_labeled.csv` — a slim table with:
  dr7objid, image_path, label (5-class), fuzzy fractions, sample split
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---- CONFIG: edit these paths for your machine ----
BASE_DIR = r"D:\galaxymorpho"

CATALOG_PATH = f"{BASE_DIR}/gz2_hart16.csv"
MAPPING_PATH = f"{BASE_DIR}/gz2_filename_mapping.csv"   # download separately from Zenodo
IMAGES_DIR   = f"{BASE_DIR}/images_gz2"                  # folder you unzipped images_gz2.zip into
OUTPUT_CSV   = f"{BASE_DIR}/gz2_labeled.csv"

CLASS_NAMES = ["disc", "spiral", "elliptical", "round", "irregular"]

# Columns we actually need from the 231-column catalog (keeps memory sane).
# These map directly onto the Galaxy Zoo decision tree used by Gauthier et al.:
#   Q1 (t01) smooth vs features vs star/artifact  -- the first fork
#   Q2 (t02) edge-on                              -- disc vs spiral fork
#   Q4 (t04) spiral pattern present                -- disc vs spiral fork
#   Q6 (t06) something odd                        -- catches "irregular"
#   Q7 (t07) how round                            -- elliptical vs round fork
KEEP_COLS = [
    "dr7objid", "ra", "dec", "sample", "gz2_class",
    "total_classifications",
    "t01_smooth_or_features_a01_smooth_debiased",
    "t01_smooth_or_features_a02_features_or_disk_debiased",
    "t01_smooth_or_features_a03_star_or_artifact_debiased",
    "t02_edgeon_a04_yes_debiased",
    "t02_edgeon_a05_no_debiased",
    "t04_spiral_a08_spiral_debiased",
    "t04_spiral_a09_no_spiral_debiased",
    "t06_odd_a14_yes_debiased",
    "t06_odd_a15_no_debiased",
    "t07_rounded_a16_completely_round_debiased",
    "t07_rounded_a17_in_between_debiased",
    "t07_rounded_a18_cigar_shaped_debiased",
]


def load_catalog() -> pd.DataFrame:
    # compression="infer" auto-detects gzip vs plain CSV from the file's
    # actual content/extension, instead of assuming gzip -- avoids the
    # "Not a gzipped file" error if you downloaded/extracted it as plain CSV.
    df = pd.read_csv(CATALOG_PATH, compression="infer", usecols=KEEP_COLS)
    print(f"Loaded catalog: {df.shape[0]:,} rows")
    return df


def load_mapping() -> pd.DataFrame:
    """gz2_filename_mapping.csv columns: objid, sample, asset_id"""
    m = pd.read_csv(MAPPING_PATH)
    m = m.rename(columns={"objid": "dr7objid"})
    print(f"Loaded filename mapping: {m.shape[0]:,} rows")
    return m


def _classify_row(row) -> str:
    """
    Reproduces the Galaxy Zoo decision tree used by Gauthier, Jain & Noordeh
    (2016) to turn the 37 raw GZ questions into 5 galaxy classes: disc,
    spiral, elliptical, round, irregular. Uses the DEBIASED vote fractions (not
    raw gz2_class strings) so the same fractions double as your fuzzy ground
    truth later in Steps 3/7.

    Decision tree:
      Q1 winner = star/artifact                -> irregular
      Q6 says "odd" AND Q1 winner != features   -> irregular
      Q1 winner = smooth  -> Q7: completely_round -> round, else -> elliptical
      Q1 winner = features -> Q2 edge-on -> disc
                            -> else Q4 spiral -> spiral, else -> disc
    """
    a01 = row["t01_smooth_or_features_a01_smooth_debiased"]
    a02 = row["t01_smooth_or_features_a02_features_or_disk_debiased"]
    a03 = row["t01_smooth_or_features_a03_star_or_artifact_debiased"]
    t01_winner = np.argmax([a01, a02, a03])  # 0=smooth, 1=features, 2=artifact

    if t01_winner == 2:
        return "irregular"

    odd_yes = row["t06_odd_a14_yes_debiased"]
    odd_no = row["t06_odd_a15_no_debiased"]
    if odd_yes > odd_no and t01_winner != 1:
        return "irregular"

    if t01_winner == 0:  # smooth
        round_ = row["t07_rounded_a16_completely_round_debiased"]
        inbetween = row["t07_rounded_a17_in_between_debiased"]
        cigar = row["t07_rounded_a18_cigar_shaped_debiased"]
        return "round" if round_ >= inbetween and round_ >= cigar else "elliptical"

    # t01_winner == 1: features/disk
    edge_yes = row["t02_edgeon_a04_yes_debiased"]
    edge_no = row["t02_edgeon_a05_no_debiased"]
    if edge_yes > edge_no:
        return "disc"

    spiral = row["t04_spiral_a08_spiral_debiased"]
    no_spiral = row["t04_spiral_a09_no_spiral_debiased"]
    return "spiral" if spiral > no_spiral else "disc"


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crisp 5-class label:  disc, spiral, elliptical, round, irregular
      -> built from debiased vote fractions via the decision tree above,
         NOT from the raw gz2_class string (that string is far more granular,
         818 unique values in this catalog -- too fine-grained to train on
         directly, and the debiased fractions are the more principled source
         since they've already been corrected for redshift/size bias).
    Fuzzy label:  use debiased vote fractions directly as soft targets.
      This is the "ground truth fuzziness" you'll validate your FCM
      membership scores against in Step 3/7.
    """
    df = df.copy()
    df["label"] = df.apply(_classify_row, axis=1)

    # Fuzzy membership targets: raw debiased probabilities per relevant axis.
    # Not a full 5-way soft label (GZ's tree structure means these don't all
    # sum to 1 across all 5 classes at once) but they're exactly what you
    # need for the elliptical<->round and disc<->spiral fuzziness checks.
    df["fuzzy_smooth_score"] = df["t01_smooth_or_features_a01_smooth_debiased"]
    df["fuzzy_features_score"] = df["t01_smooth_or_features_a02_features_or_disk_debiased"]
    df["fuzzy_spiral_score"] = df["t04_spiral_a08_spiral_debiased"]
    df["fuzzy_round_score"] = df["t07_rounded_a16_completely_round_debiased"]

    print(f"Labeled {df.shape[0]:,} rows")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True).round(3))
    return df


def attach_image_paths(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(mapping[["dr7objid", "asset_id"]], on="dr7objid", how="inner")
    print(f"After matching to available images: {merged.shape[0]:,} rows "
          f"({df.shape[0] - merged.shape[0]:,} catalog rows had no image)")

    merged["image_path"] = merged["asset_id"].astype(str).apply(
        lambda a: str(Path(IMAGES_DIR) / f"{a}.jpg")
    )
    return merged


def main():
    df = load_catalog()
    df = build_labels(df)
    mapping = load_mapping()
    df = attach_image_paths(df, mapping)

    # Sanity check: confirm a sample of image files actually exist on disk
    exists = df["image_path"].head(200).apply(lambda p: Path(p).exists())
    print(f"Sanity check: {exists.sum()}/200 sampled image files found on disk. "
          f"If this is 0, check IMAGES_DIR path / that you unzipped images_gz2.zip.")

    out_cols = [
        "dr7objid", "image_path", "label",
        "fuzzy_smooth_score", "fuzzy_features_score", "fuzzy_spiral_score", "fuzzy_round_score",
        "sample", "ra", "dec",
    ]
    df[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} with {df.shape[0]:,} rows")


if __name__ == "__main__":
    main()