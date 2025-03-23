# config.py

import os

# データセットのパス
dataset_dir = 'data/fruit30_split'

# モデル保存のパス
checkpoint_dir = '/content/drive/My Drive/Colab Notebooks/checkpoint'

# ハイパーパラメータ設定
BATCH_SIZE = 32
EPOCHS = 30

# その他の設定項目
LEARNING_RATE = 0.001

# 最良モデル保存のパス形式
BEST_MODEL_PATH = os.path.join(checkpoint_dir, 'best-{:.3f}.pth')

# 現在の最良精度
BEST_TEST_ACCURACY = 0.0  # 初期値は 0 で、最良のテスト精度を保存するために使用


