from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(".")
CLEAN_DIR = PROJECT_ROOT / "outputs" / "cleaned_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "folds"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CLEAN_DIR / "train_clean.csv")

print("Loaded cleaned train data:", df.shape)

# Ensure correct types
df["label"] = df["label"].astype(int)


# =========================
# CREATE FOLDS
# =========================
df["fold"] = -1

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
    df.loc[val_idx, "fold"] = fold


# =========================
# VALIDATION CHECK
# =========================
print("\nFold distribution:")
print(df["fold"].value_counts().sort_index())

print("\nLabel distribution per fold:")
for fold in range(N_SPLITS):
    print(f"\nFold {fold}")
    print(df[df["fold"] == fold]["label"].value_counts(normalize=True))


# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_DIR / "train_folds.csv", index=False)

print("\nSaved to:", OUTPUT_DIR / "train_folds.csv")