import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

# 設定設備 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義數據增強和預處理步驟
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 預處理參數來自ImageNet
])

# 假設資料夾結構為：data/train 和 data/val
train_dir = 'data/train'
val_dir = 'data/val'

# 加載數據集
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加載預訓練的ResNet18模型
model = models.resnet18(pretrained=True)

# 修改最後一層的全連接層，因為我們的分類數目是10
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 移動到GPU（如果可用）
model = model.to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練過程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    # 訓練階段
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向傳播和優化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 計算準確度
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels)
        total_preds += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_preds.double() / total_preds
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # 驗證階段
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

    val_accuracy = correct_preds.double() / total_preds
    print(f"Validation Accuracy: {val_accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), 'resnet_bird_classifier.pth')