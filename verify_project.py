from pathlib import Path

# ==========================================================
# Research Framework Verification
# ==========================================================

ROOT = Path(__file__).resolve().parent

print("=" * 70)
print("Research Framework Verification")
print("=" * 70)

required_dirs = [
    "configs",
    "models",
    "datasets",
    "cam",
    "metrics",
    "utils",
    "raw_data",
    "dataset",
    "outputs",
]

required_files = [
    "train.py",
    "evaluate.py",
    "generate_heatmaps.py",
    "evaluate_localization.py",
    "compare_backbones.py",
    "main.py",
]

print("\nChecking directories...\n")

all_ok = True

for folder in required_dirs:

    path = ROOT / folder

    if path.exists():
        print(f"[OK] {folder}")
    else:
        print(f"[Missing] {folder}")
        all_ok = False

print("\nChecking python files...\n")

for file in required_files:

    path = ROOT / file

    if path.exists():
        print(f"[OK] {file}")
    else:
        print(f"[Missing] {file}")
        all_ok = False

print("\nChecking AutoSplice dataset...\n")

dataset_paths = [
    "raw_data/AutoSplice/Authentic",
    "raw_data/AutoSplice/Forged_JPEG100",
    "raw_data/AutoSplice/Mask",
    "raw_data/AutoSplice/Caption",
]

for p in dataset_paths:

    path = ROOT / p

    if path.exists():
        print(f"[OK] {p}")
    else:
        print(f"[Missing] {p}")
        all_ok = False

print("\nChecking prepared dataset...\n")

prepared = [
    "dataset/train/real",
    "dataset/train/edited",
    "dataset/val/real",
    "dataset/val/edited",
    "dataset/test/real",
    "dataset/test/edited",
    "dataset/masks/train/edited",
    "dataset/masks/val/edited",
    "dataset/masks/test/edited",
]

for p in prepared:

    path = ROOT / p

    if path.exists():
        print(f"[OK] {p}")
    else:
        print(f"[Missing] {p}")
        all_ok = False

print("\nChecking config files...\n")

configs = [
    "configs/resnet50.yaml",
    "configs/efficientnet_b0.yaml",
    "configs/convnext_tiny.yaml",
    "configs/swin_tiny.yaml",
    "configs/vit_b16.yaml",
]

for c in configs:

    path = ROOT / c

    if path.exists():
        print(f"[OK] {c}")
    else:
        print(f"[Missing] {c}")
        all_ok = False

print("\nCreating output folders...\n")

output_dirs = [
    "outputs/checkpoints",
    "outputs/logs",
    "outputs/heatmaps",
    "outputs/evaluation",
    "outputs/figures",
    "outputs/reports",
]

for folder in output_dirs:

    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)

    print(f"[Created] {folder}")

print("\n")

if all_ok:

    print("=" * 70)
    print("PROJECT VERIFICATION PASSED")
    print("=" * 70)

else:

    print("=" * 70)
    print("PROJECT VERIFICATION FINISHED WITH WARNINGS")
    print("=" * 70)