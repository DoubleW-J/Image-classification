# config.py

import torch

# トレーニング設定
cuda = torch.cuda.is_available()  # GPUが利用可能か自動検出
pre_train = True  # 事前学習モデルを使用するか
CosineLR = True  # コサインアニーリング学習率を使用するか

# ハイパーパラメータ
lr = 1e-3  # 学習率
Batch_size = 32  # バッチサイズ
Init_Epoch = 0  # 初期エポック数
Fin_Epoch = 100  # 総エポック数

# モデル関連
num_classes = 2  # クラス数
pretrained_model_path = r'logs\resnet50_imagenet.pth'  # 事前学習モデルのパス


# データセットのパス
dataset_dir = r'D:\Download\dataset\helmet_split\helmet_classification'
train_path = dataset_dir + r'\train'  # 訓練データのパス
test_path = dataset_dir + r'\val'  # 検証データのパス
