import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

# 自作モジュールと設定のインポート
import config
from nets.resnet50 import ResNet, Bottleneck

def main():
    # --- 1. 設定の読み込み ---
    batch_size = config.Config.BATCH_SIZE
    test_path = config.Config.DATASET_DIR  # 検証用データのルートパス
    log_dir = config.Config.LOG_DIR
    num_classes = config.Config.NUM_CLASSES
    
    # ログディレクトリ内のファイル一覧を取得
    if not os.path.exists(log_dir):
        print(f"[Error] ログディレクトリが見つかりません: {log_dir}")
        return
    file_list = [f for f in os.listdir(log_dir) if f.endswith('.pth')]

    # --- 2. データセットの準備 (ループの外で行う) ---
    # 推論用の前処理（訓練時の設定と合わせるのが鉄則）
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ImageFolderを使用して検証データ（val）をロード
    val_data_path = os.path.join(test_path, 'val')
    if not os.path.exists(val_data_path):
        print(f"[Error] 検証データが見つかりません: {val_data_path}")
        return

    test_dataset = datasets.ImageFolder(val_data_path, transform=test_transform)
    gen_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # モデルの基本構造を定義
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model = model.cuda()

    print(f"[*] 評価開始: 対象ファイル数 {len(file_list)}")
    print("-" * 50)

    # --- 3. モデルごとの評価ループ ---
    for file in file_list:
        path = os.path.join(log_dir, file)

        # 重みのロード
        try:
            model.load_state_dict(torch.load(path))
        except Exception as e:
            print(f"[Skip] {file} のロードに失敗しました: {e}")
            continue

        model.eval()
        test_correct = 0

        # 推論実行
        with torch.no_grad(): # 勾配計算をオフにしてメモリ節約
            for data in gen_test:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()

        # 精度の算出と表示
        accuracy = 100.0 * test_correct / len(test_dataset)
        print(f"ファイル名: {file}")
        print(f"正解率 (Accuracy): {accuracy:.3f}%")
        print("-" * 50)

if __name__ == "__main__":
    main()