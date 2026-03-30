from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import imagehash


PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"

POSSIBLE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_images_dir() -> Path:
    candidates = [
        DATA_DIR / "images",
        DATA_DIR / "images_final_sample",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    available_dirs = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()) if DATA_DIR.exists() else []
    raise FileNotFoundError(
        f"No image directory found under {DATA_DIR}. Tried: {candidates}. Available directories: {available_dirs}"
    )


def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("\nTrain columns:", list(train_df.columns))
    print("Test columns :", list(test_df.columns))

    return train_df, test_df


def standardize_train_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()

    if "ground_truth" not in train_df.columns:
        raise ValueError("train.csv must contain a 'ground_truth' column")

    label_map = {
        "authentic": 0,
        "ai_generated": 1,
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
    }

    train_df["label"] = train_df["ground_truth"].map(label_map)

    unmapped = train_df[train_df["label"].isna()]
    if len(unmapped) > 0:
        print("\nWARNING: Unmapped labels found:")
        print(unmapped["ground_truth"].value_counts(dropna=False))

    return train_df


def find_image_path(image_id: str, images_dir: Path) -> str | None:
    image_id = str(image_id).strip()

    # Case 1: image_id already includes extension
    direct_candidate = images_dir / image_id
    if direct_candidate.exists():
        return str(direct_candidate)

    # Case 2: image_id is base name without extension
    for ext in POSSIBLE_EXTENSIONS:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return str(candidate)

    # Case 3: fallback loose match
    matches = list(images_dir.glob(f"{image_id}*"))
    if len(matches) > 0:
        return str(matches[0])

    return None


def add_image_paths(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["image_path"] = df["image_id"].apply(lambda x: find_image_path(x, images_dir))
    df["file_exists"] = df["image_path"].notna()
    return df


def get_all_image_files(images_dir: Path) -> list[Path]:
    files = []
    for ext in POSSIBLE_EXTENSIONS:
        files.extend(images_dir.glob(f"*{ext}"))
        files.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def safe_open_image(path: str) -> dict:
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            fmt = img.format

        return {
            "status": "ok",
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
        }
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return {
            "status": "corrupt",
            "error": str(e),
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
        }


def compute_file_hash(path: str, chunk_size: int = 8192) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_perceptual_hash(path: str) -> str | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return str(imagehash.phash(img))
    except Exception:
        return None


# =========================
# AUDIT STEPS
# =========================
def basic_id_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    summary = {}

    train_dupes = train_df[train_df.duplicated(subset=["image_id"], keep=False)].sort_values("image_id")
    test_dupes = test_df[test_df.duplicated(subset=["image_id"], keep=False)].sort_values("image_id")

    overlap_ids = sorted(set(train_df["image_id"].astype(str)).intersection(set(test_df["image_id"].astype(str))))

    train_dupes.to_csv(OUTPUT_DIR / "train_duplicate_ids.csv", index=False)
    test_dupes.to_csv(OUTPUT_DIR / "test_duplicate_ids.csv", index=False)

    pd.DataFrame({"image_id": overlap_ids}).to_csv(OUTPUT_DIR / "train_test_overlap_ids.csv", index=False)

    summary["train_duplicate_id_rows"] = int(len(train_dupes))
    summary["test_duplicate_id_rows"] = int(len(test_dupes))
    summary["train_test_overlap_count"] = int(len(overlap_ids))

    return summary


def file_existence_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = add_image_paths(train_df, IMAGES_DIR)
    test_df = add_image_paths(test_df, IMAGES_DIR)

    missing_train = train_df[~train_df["file_exists"]]
    missing_test = test_df[~test_df["file_exists"]]

    missing_train.to_csv(OUTPUT_DIR / "missing_train_files.csv", index=False)
    missing_test.to_csv(OUTPUT_DIR / "missing_test_files.csv", index=False)

    all_files = get_all_image_files(IMAGES_DIR)

    # Compare using actual filenames and stems
    physical_names = {p.name for p in all_files}
    physical_stems = {p.stem for p in all_files}

    referenced_ids = set(train_df["image_id"].astype(str)).union(set(test_df["image_id"].astype(str)))

    unreferenced_files = []
    for p in all_files:
        if p.name not in referenced_ids and p.stem not in referenced_ids:
            unreferenced_files.append(p.name)

    pd.DataFrame({"image_file": sorted(unreferenced_files)}).to_csv(
        OUTPUT_DIR / "unreferenced_image_files.csv", index=False
    )

    summary = {
        "train_missing_files": int(len(missing_train)),
        "test_missing_files": int(len(missing_test)),
        "physical_image_files_found": int(len(all_files)),
        "unique_physical_image_names_found": int(len(physical_names)),
        "unique_physical_image_stems_found": int(len(physical_stems)),
        "unreferenced_image_files": int(len(unreferenced_files)),
    }

    return train_df, test_df, summary


def image_integrity_and_metadata(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [
            train_df[["image_id", "image_path"]].assign(split="train"),
            test_df[["image_id", "image_path"]].assign(split="test"),
        ],
        ignore_index=True,
    )

    combined = combined[combined["image_path"].notna()].drop_duplicates(subset=["image_path"]).copy()

    records = []
    for _, row in tqdm(combined.iterrows(), total=len(combined), desc="Reading images"):
        info = safe_open_image(row["image_path"])
        records.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "image_path": row["image_path"],
                **info,
            }
        )

    meta_df = pd.DataFrame(
        records,
        columns=["image_id", "split", "image_path", "status", "error", "width", "height", "mode", "format"],
    )
    meta_df.to_csv(OUTPUT_DIR / "image_metadata.csv", index=False)

    if meta_df.empty:
        print("No image files were resolved from train/test rows. Skipping integrity report.")
        corrupt_df = meta_df.copy()
    else:
        corrupt_df = meta_df[meta_df["status"] != "ok"].copy()
    corrupt_df.to_csv(OUTPUT_DIR / "corrupt_files.csv", index=False)

    return meta_df


def duplicate_checks(meta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok_df = meta_df[meta_df["status"] == "ok"].copy()

    file_hashes = []
    phashes = []

    for path in tqdm(ok_df["image_path"], total=len(ok_df), desc="Hashing images"):
        file_hashes.append(compute_file_hash(path))
        phashes.append(compute_perceptual_hash(path))

    ok_df["file_md5"] = file_hashes
    ok_df["phash"] = phashes

    exact_dupes = ok_df[ok_df.duplicated(subset=["file_md5"], keep=False)].sort_values("file_md5")
    near_dupes = ok_df[ok_df.duplicated(subset=["phash"], keep=False)].sort_values("phash")

    exact_dupes.to_csv(OUTPUT_DIR / "exact_duplicate_images.csv", index=False)
    near_dupes.to_csv(OUTPUT_DIR / "near_duplicate_images.csv", index=False)

    ok_df.to_csv(OUTPUT_DIR / "image_metadata_with_hashes.csv", index=False)

    return exact_dupes, near_dupes


def metadata_summary(meta_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    ok_df = meta_df[meta_df["status"] == "ok"].copy()

    summary = {
        "images_successfully_read": int(len(ok_df)),
        "corrupt_or_unreadable_images": int((meta_df["status"] != "ok").sum()),
        "format_distribution": ok_df["format"].value_counts(dropna=False).to_dict(),
        "mode_distribution": ok_df["mode"].value_counts(dropna=False).to_dict(),
    }

    if len(ok_df) > 0:
        summary["width_stats"] = {
            "min": int(ok_df["width"].min()),
            "max": int(ok_df["width"].max()),
            "mean": float(ok_df["width"].mean()),
            "median": float(ok_df["width"].median()),
        }
        summary["height_stats"] = {
            "min": int(ok_df["height"].min()),
            "max": int(ok_df["height"].max()),
            "mean": float(ok_df["height"].mean()),
            "median": float(ok_df["height"].median()),
        }

    if "ground_truth" in train_df.columns:
        summary["train_label_distribution"] = train_df["ground_truth"].value_counts(dropna=False).to_dict()

    return summary


def save_final_summary(summary: dict) -> None:
    with open(OUTPUT_DIR / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=4)


def main():
    ensure_output_dir()

    train_df, test_df = load_csvs()
    train_df = standardize_train_labels(train_df)

    summary = {}
    summary["train_rows"] = int(len(train_df))
    summary["test_rows"] = int(len(test_df))

    print("\nStep 1: Checking duplicate IDs and train/test overlap...")
    summary.update(basic_id_checks(train_df, test_df))

    print("\nStep 2: Checking file existence...")
    train_df, test_df, exist_summary = file_existence_checks(train_df, test_df)
    summary.update(exist_summary)

    train_df.to_csv(OUTPUT_DIR / "train_with_paths.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_with_paths.csv", index=False)

    print("\nStep 3: Checking image integrity and metadata...")
    meta_df = image_integrity_and_metadata(train_df, test_df)

    print("\nStep 4: Checking exact and near duplicates...")
    exact_dupes, near_dupes = duplicate_checks(meta_df)

    summary["exact_duplicate_image_rows"] = int(len(exact_dupes))
    summary["near_duplicate_image_rows"] = int(len(near_dupes))

    summary.update(metadata_summary(meta_df, train_df))

    save_final_summary(summary)

    print("\nAudit completed.")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
