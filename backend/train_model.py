"""Baseline ResNet18 training script.
Expected folders: Dataset/ASD/ and Dataset/non-ASD/.
Use participant-level splits for credible evaluation; this demo split is image-level.
"""
from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Dataset"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.is_dir():
    raise SystemExit(f"Dataset folder not found: {DATA_DIR}")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
if len(dataset.classes) != 2:
    raise SystemExit(f"Expected exactly two class folders; found: {dataset.classes}")
counts = [sum(1 for _, label in dataset.samples if label == i) for i in range(2)]
if min(counts) < 2:
    raise SystemExit("Each class needs at least two valid images.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
if not train_size or not val_size:
    raise SystemExit("Not enough images for train/validation split.")
train_data, val_data = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_accuracy = -1.0

print("Class mapping:", dataset.class_to_idx)
print(f"Images: {len(dataset)} | Train: {train_size} | Validation: {val_size}")
print("Device:", device)
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / max(total, 1)
    print(f"Epoch {epoch + 1}/10 | Validation accuracy: {accuracy:.3f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.cpu().state_dict(), MODEL_DIR / "asd_resnet18.pth")
        model.to(device)

with (MODEL_DIR / "classes.json").open("w", encoding="utf-8") as f:
    json.dump(dataset.class_to_idx, f, indent=2)
print(f"Saved baseline model (validation accuracy={best_accuracy:.3f}) to {MODEL_DIR}")
