import torch
import torchvision
import cv2
import time
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from google.colab.patches import cv2_imshow

# 自作モジュールと設定のインポート
from config import Config
from nets.resnet50 import ResNet, Bottleneck

def main():
    # --- 1. 環境設定 & モデルの準備 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 最新のベストモデル、または指定した重みファイルを選択
    # ここでは例として config で指定したパスを使用
    model_path = os.path.join(Config.LOG_DIR, "best_model_acc_97.68.pth") 
    
    if not os.path.exists(model_path):
        print(f"[Error] モデルファイルが見つかりません: {model_path}")
        return

    # モデル構築（2クラス分類設定）
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 2. テストデータの準備 ---
    # 推論用の前処理（Normalizeは訓練時と同じ設定）
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # 表示用に正規化（Normalize）なしのデータセットも用意するか、後で逆計算する
    # ここではシンプルに検証用フォルダから読み込み
    test_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, 'val'), transform=test_transform)
    
    batch_size = int(input('プレビューする画像枚数を入力してください: '))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # クラス名の定義
    class_names = ["No_Helmet (未着用)", "Helmet (着用)"]

    print(f"\n[*] 推論開始（デバイス: {device}）...")

    try:
        while True:
            # バッチデータの取得
            images, labels = next(iter(test_loader))
            inputs = images.to(device)

            # 推論実行と時間計測
            start_time = time.time()
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            infer_time = time.time() - start_time

            # --- 3. 結果の表示 ---
            # torchvision.utils.make_grid を使って1枚の画像にまとめる
            grid_img = torchvision.utils.make_grid(images, nrow=batch_size)
            # (C, H, W) -> (H, W, C) に変換し、OpenCV用に色空間を調整
            img_np = grid_img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            print("-" * 30)
            print(f"推論時間: {infer_time:.4f}秒")
            print(f"判定結果: {[class_names[p] for p in preds.cpu().numpy()]}")
            print(f"正解ラベル: {[class_names[l] for l in labels.numpy()]}")

            # 画像表示
            cv2_imshow(img_bgr)

            cont = input("\n次のバッチを表示しますか？ (y/n): ")
            if cont.lower() != 'y':
                break

    except StopIteration:
        print("データセットの末尾に到達しました。")

if __name__ == "__main__":
    main()