import torch
from torch import nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import torchvision.transforms as T
from datasets import load_dataset
from tqdm import tqdm

from vit import VIT


def transform_fn(sample):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = [transform(img) for img in sample["image"]]
    lbl = torch.tensor(sample["label"], dtype=torch.long)

    return {"image": img, "label": lbl}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = load_dataset(
        "benjamin-paine/imagenet-1k-128x128", split="train"
    ).with_transform(transform_fn)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = VIT(
        patch_size=8,
        d_emb=128,
        num_heads=8,
        seq_len=256,
        layers=8,
    )
    model = nn.Sequential(
        model, torch.nn.Linear(128, 1000)  # Assuming 1000 classes for ImageNet
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=95)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        X, y = batch["image"].to(device), batch["label"].to(device)

        pred = model(X)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

    scheduler.step()


if __name__ == "__main__":
    main()
