import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==================================================
# PATHS (MATCH YOUR DATASET EXACTLY)
# ==================================================
DATA_DIR = Path("data")

# Images are inside:
# data/BCN_20k_train/bcn_20k_train/*.jpg
IMAGES_ROOT = DATA_DIR / "BCN_20k_train" / "bcn_20k_train"

# CSV file
CSV_PATH = DATA_DIR / "bcn_20k_train.csv"

# Output balanced dataset
OUT = DATA_DIR / "balanced_dataset"

# ==================================================
# LOAD CSV
# ==================================================
df = pd.read_csv(CSV_PATH)

# ==================================================
# MAP DIAGNOSIS CODES → FINAL CLASSES
# ==================================================
# MEL  → Melanoma
# BCC  → Basal Cell Carcinoma
# NV   → Benign
# BKL  → Benign
# SCC, AK → excluded
mapping = {
    "MEL": "melanoma",
    "BCC": "basal_cell_carcinoma",
    "NV": "benign",
    "BKL": "benign",
}

df = df[df["diagnosis"].isin(mapping)]
df["label"] = df["diagnosis"].map(mapping)

print("Class distribution after mapping:")
print(df["label"].value_counts())

# ==================================================
# BALANCE DATASET
# ==================================================
SAMPLES_PER_CLASS = 1500
dfs = []

for cls in df["label"].unique():
    cls_df = df[df["label"] == cls]
    cls_df = cls_df.sample(
        n=min(SAMPLES_PER_CLASS, len(cls_df)),
        random_state=42
    )
    dfs.append(cls_df)

data = pd.concat(dfs, ignore_index=True)

# ==================================================
# TRAIN / VAL / TEST SPLIT
# ==================================================
train, temp = train_test_split(
    data,
    test_size=0.2,
    stratify=data["label"],
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.5,
    stratify=temp["label"],
    random_state=42
)

splits = {
    "train": train,
    "val": val,
    "test": test
}

# ==================================================
# COPY IMAGES
# ==================================================
missing = 0
copied = 0

for split, split_df in splits.items():
    for cls in split_df["label"].unique():
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    for _, row in split_df.iterrows():
        img_name = row["bcn_filename"]

        src = IMAGES_ROOT / img_name
        dst = OUT / split / row["label"] / img_name

        if src.exists():
            shutil.copy(src, dst)
            copied += 1
        else:
            missing += 1

print(f"✅ Images copied: {copied}")
print(f"⚠️ Missing images skipped: {missing}")
print("✅ Balanced dataset created successfully")
