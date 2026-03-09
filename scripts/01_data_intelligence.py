"""
scripts/01_data_intelligence.py

Data Intelligence — Scans the raw JPEG cardiac MRI dataset and prints a
structural overview including patient counts, series counts, slice counts,
and class distributions.

Data format
-----------
data/raw/
    {Normal|Sick}/
        Directory_N/          ← one patient
            series_X-Body/    ← one MRI series
                imgNNNN-z.jpg ← one 2-D slice

Usage
-----
    python scripts/01_data_intelligence.py
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/raw")
CLASSES = ["Normal", "Sick"]


def scan_jpeg_structure():
    print(f"🔍 Scanning {DATA_PATH}...\n")

    records = []
    class_patient_counts = defaultdict(int)
    class_series_counts  = defaultdict(int)
    class_image_counts   = defaultdict(int)

    for class_label in CLASSES:
        class_dir = DATA_PATH / class_label
        if not class_dir.exists():
            print(f"⚠️  Skipping missing class folder: {class_dir}")
            continue

        patient_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
        class_patient_counts[class_label] = len(patient_dirs)

        for patient_dir in patient_dirs:
            patient_id = f"{class_label}/{patient_dir.name}"
            series_dirs = sorted([s for s in patient_dir.iterdir() if s.is_dir()])

            for series_dir in series_dirs:
                jpg_files = list(series_dir.glob("*.jpg"))
                n_imgs = len(jpg_files)
                class_series_counts[class_label]  += 1
                class_image_counts[class_label]   += n_imgs
                records.append({
                    "patient_id":   patient_id,
                    "class":        class_label,
                    "series_path":  str(series_dir),
                    "image_count":  n_imgs,
                })

    # ---------- Summary ---------------------------------------------------
    total_patients = sum(class_patient_counts.values())
    total_series   = sum(class_series_counts.values())
    total_images   = sum(class_image_counts.values())

    print(f"{'Class':<10}  {'Patients':>9}  {'Series':>8}  {'Images':>9}")
    print("-" * 44)
    for cls in CLASSES:
        print(
            f"{cls:<10}  {class_patient_counts[cls]:>9,}  "
            f"{class_series_counts[cls]:>8,}  "
            f"{class_image_counts[cls]:>9,}"
        )
    print("-" * 44)
    print(
        f"{'TOTAL':<10}  {total_patients:>9,}  "
        f"{total_series:>8,}  {total_images:>9,}"
    )

    print(f"\n📊 Found {total_patients} unique patients across {total_series} series")
    print(f"   Total JPG slices : {total_images:,}")

    # ---------- Save metadata ---------------------------------------------
    metadata_dir = Path("data/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(metadata_dir / "patient_summary.csv", index=False)
    print("\n✅ Saved patient summary to data/metadata/patient_summary.csv")

    # Patient-level summary
    patient_df = (
        df.groupby(["patient_id", "class"])
          .agg(series_count=("series_path", "count"),
               total_images=("image_count", "sum"))
          .reset_index()
    )
    patient_df.to_csv(metadata_dir / "patient_level_summary.csv", index=False)
    print("✅ Saved patient-level summary to data/metadata/patient_level_summary.csv")


if __name__ == "__main__":
    scan_jpeg_structure()