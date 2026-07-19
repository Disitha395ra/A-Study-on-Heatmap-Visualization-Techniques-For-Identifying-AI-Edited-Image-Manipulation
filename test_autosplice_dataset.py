from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets.collate import autosplice_collate_fn

from datasets.autosplice_dataset import AutoSpliceDataset
from datasets.transforms import (
    get_train_image_transform,
    get_eval_image_transform,
    get_mask_transform,
)


ROOT = Path(__file__).resolve().parent


def main():
    print("=" * 70)
    print("AutoSpliceDataset Loader Test")
    print("=" * 70)

    train_dataset = AutoSpliceDataset(
        project_root=ROOT,
        split="train",
        image_transform=get_train_image_transform(224),
        mask_transform=get_mask_transform(224),
        image_size=224,
    )

    test_dataset = AutoSpliceDataset(
        project_root=ROOT,
        split="test",
        image_transform=get_eval_image_transform(224),
        mask_transform=get_mask_transform(224),
        image_size=224,
    )

    print("Train size:", len(train_dataset))
    print("Test size :", len(test_dataset))

    sample = train_dataset[0]

    print("\nSingle sample:")
    print("image shape :", sample["image"].shape)
    print("label       :", sample["label"])
    print("mask shape  :", sample["mask"].shape)
    print("image name  :", sample["image_name"])
    print("class name  :", sample["class_name"])
    print("is edited   :", sample["is_edited"])
    print("caption     :", sample["caption"])

    loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=autosplice_collate_fn,
)

    batch = next(iter(loader))

    print("\nBatch:")
    print("images:", batch["image"].shape)
    print("labels:", batch["label"].shape)
    print("masks :", batch["mask"].shape)
    print("names :", batch["image_name"][:3])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    print("\nMoved to device:", images.device, labels.device)

    print("\nStep 13D completed successfully.")


if __name__ == "__main__":
    main()