import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = r"D:\galaxymorpho"

CATALOG_PATH = f"{BASE_DIR}/gz2_hart16.csv"
MAPPING_PATH = f"{BASE_DIR}/gz2_filename_mapping.csv"   
IMAGES_DIR   = f"{BASE_DIR}/images_gz2"                 
OUTPUT_CSV   = f"{BASE_DIR}/gz2_labeled.csv"

CLASS_NAMES = ["disc", "spiral", "elliptical", "round", "irregular"]

                  -- catches "irregular"

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
    
    df = df.copy()
    df["label"] = df.apply(_classify_row, axis=1)

    
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
