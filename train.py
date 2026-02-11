import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
from tqdm import tqdm

# モジュールのインポート
from config import Config
from nets.resnet50 import Bottleneck, ResNet

# ==========================================
# データの前処理 (Data Preprocessing)
# ==========================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 学習ループ関数 (Training Function)
# ==========================================
def fit_one_epoch(net, criterion, optimizer, epoch, train_loader, val_loader, device, best_acc, history):
    # 訓練
    net.train()
    total_train_loss = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{Config.EPOCHS} [Train]') as pbar:
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.update(1)

    # 検証
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = 100.0 * correct / total
    history['acc'].append(acc)
    
    # 最高精度の保存
    if acc > best_acc:
        best_acc = acc
        save_path = os.path.join(Config.LOG_DIR, f'best_model_acc_{acc:.2f}.pth')
        torch.save(net.state_dict(), save_path)
        print(f"[*] モデルを保存しました: {save_path}")
    
    return best_acc

if __name__ == "__main__":
    device = torch.device('cuda' if Config.CUDA else 'cpu')

    # 設定ファイルに基づいたデータロード
    train_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, 'val'), test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # モデル構築
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=Config.NUM_CLASSES).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    best_accuracy = 0.0
    history = {'acc': []}

    for epoch in range(Config.EPOCHS):
        best_accuracy = fit_one_epoch(model, criterion, optimizer, epoch, train_loader, val_loader, device, best_accuracy, history)
        scheduler.step()