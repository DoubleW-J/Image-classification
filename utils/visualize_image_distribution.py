import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# データファイルを読み込む
df = pd.read_csv('Data_volume_statistics.csv')
print(df.shape)  # データの形を表示
print(df.head())  # データの先頭を表示

# 可視化する特徴を指定
feature = 'total'  # 'total', 'trainset', 'testset' のいずれかを選択

# 指定した特徴でデータをソート
df = df.sort_values(by=feature, ascending=False)
print(df.head())  # ソートされたデータの先頭を表示

# 各カテゴリの画像数の棒グラフを描画
plt.figure(figsize=(22, 7))
x = df['class']
y = df[feature]

# 棒グラフを描く
plt.bar(x, y, facecolor='#1f77b4', edgecolor='k')

# グラフのラベルとフォーマットを設定
plt.xticks(rotation=90)
plt.tick_params(labelsize=15)
plt.xlabel('Category', fontsize=20)  # ラベルを英語に変更
plt.ylabel('Number of Images', fontsize=20)  # ラベルを英語に変更

# グラフを表示
plt.show()

# 訓練データとテストデータの積み上げ棒グラフを描画
plt.figure(figsize=(22, 7))
x = df['class']
y1 = df['testset']
y2 = df['trainset']

width = 0.55  # 棒グラフの幅を設定

# 積み上げ棒グラフを描く
plt.bar(x, y1, width, label='Test Set')
plt.bar(x, y2, width, label='Train Set', bottom=y1)

# グラフのラベルとフォーマットを設定
plt.xlabel('Category', fontsize=20)  # ラベルを英語に変更
plt.ylabel('Number of Images', fontsize=20)  # ラベルを英語に変更
plt.tick_params(labelsize=13)  # 軸の文字サイズを設定

# 凡例を追加
plt.legend(fontsize=16)  # 凡例を英語に変更

# グラフを高解像度のPDFファイルとして保存
plt.savefig('Number_of_Images_per_Category.pdf', dpi=120, bbox_inches='tight')

# グラフを表示
plt.show()
