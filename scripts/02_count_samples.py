
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data/raw"
stats = {"Normal": 0, "Sick": 0}


def count_image_files(class_label):
    class_path = DATA_ROOT / class_label
    if not class_path.exists():
        print(f"⚠️ Skipping missing folder: {class_path}")
        return 0
    total = 0
    for directory in sorted(class_path.iterdir()):
        if not directory.is_dir():
            continue
        img_files = list(directory.rglob("*.jpg"))
        print(f"{class_label}/{directory.name}: {len(img_files):,} JPG files")
        total += len(img_files)
    return total


print("\n--- JPG File Counts ---")
for label in ["Normal", "Sick"]:
    stats[label] = count_image_files(label)

print("\n--- Summary ---")
print(f"✅ Normal images: {stats['Normal']:,}")
print(f"✅ Sick images: {stats['Sick']:,}")
print(f"✅ Total: {stats['Normal'] + stats['Sick']:,}")