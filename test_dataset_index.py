from pathlib import Path

from datasets.dataset_index import build_dataset_index


ROOT = Path(__file__).resolve().parent

for split in ["train", "val", "test"]:

    samples = build_dataset_index(ROOT, split)

    real = sum(x["label"] == 1 for x in samples)
    edited = sum(x["label"] == 0 for x in samples)

    masks = sum(x["mask_path"] is not None for x in samples)

    captions = sum(x["caption"] is not None for x in samples)

    print("=" * 60)
    print(split.upper())
    print("=" * 60)

    print("Total :", len(samples))
    print("Real  :", real)
    print("Edited:", edited)
    print("Masks :", masks)
    print("Caps  :", captions)

    print("First sample:")

    print(samples[0])