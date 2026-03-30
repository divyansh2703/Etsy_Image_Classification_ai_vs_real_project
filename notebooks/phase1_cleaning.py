from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(".")
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cleaned_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read the audit outputs
train_with_paths = pd.read_csv(AUDIT_DIR / "train_with_paths.csv")
test_with_paths = pd.read_csv(AUDIT_DIR / "test_with_paths.csv")

# Split into usable and missing
train_clean = train_with_paths[train_with_paths["file_exists"] == True].copy()
test_clean = test_with_paths[test_with_paths["file_exists"] == True].copy()

train_missing = train_with_paths[train_with_paths["file_exists"] == False].copy()
test_missing = test_with_paths[test_with_paths["file_exists"] == False].copy()

# Save
train_clean.to_csv(OUTPUT_DIR / "train_clean.csv", index=False)
test_clean.to_csv(OUTPUT_DIR / "test_clean.csv", index=False)
train_missing.to_csv(OUTPUT_DIR / "train_missing.csv", index=False)
test_missing.to_csv(OUTPUT_DIR / "test_missing.csv", index=False)

# Print summary
print("Saved cleaned files to:", OUTPUT_DIR.resolve())
print()
print("Train clean rows   :", len(train_clean))
print("Train missing rows :", len(train_missing))
print("Test clean rows    :", len(test_clean))
print("Test missing rows  :", len(test_missing))

print()
print("Train label distribution after cleaning:")
print(train_clean["label"].value_counts().sort_index())