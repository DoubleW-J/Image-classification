import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

# データセットのパスを指定する
dataset_path = r'D:\Download\dataset\fruits'
os.chdir(dataset_path)
print(os.listdir())

# 画像ファイルを読み込む
dataset_path = r"D:\Download\dataset\fruits"
os.chdir(dataset_path)

data = []
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# 各カテゴリを繰り返し処理する
for fruit in tqdm(os.listdir()):
    fruit_path = os.path.join(dataset_path, fruit)

    if not os.path.isdir(fruit_path):  
        continue  # フォルダでない内容をスキップする
    
    for file in os.listdir(fruit_path):  # そのカテゴリのすべての画像を繰り返し処理する
        if not any(file.lower().endswith(ext) for ext in valid_extensions):
            continue  # 画像ファイルでないものをスキップする

        img_path = os.path.join(fruit_path, file)  
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            print(img_path, 'Read error')
            continue

        data.append({'Category': fruit, 'Filename': file, 'Image_Width': img.shape[1], 'Image_Height': img.shape[0]})

# DataFrameを作成する
df = pd.DataFrame(data)

# 最初の数行のデータを表示する
print(df.head())

# CSVファイルに保存する
df.to_csv('fruits_dataset.csv', index=False)
print("Data has been saved to fruits_dataset.csv")

# プロットする
x = df['Image_Width']
y = df['Image_Height']

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# 密度でポイントを並べ替え、最も密なポイントが最後にプロットされるようにする
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(10,10))
# plt.figure(figsize=(12,12))
plt.scatter(x, y, c=z,  s=5, cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])

plt.tick_params(labelsize=15)

# xy_max = max(max(df['Image_Width']), max(df['Image_Height']))
# plt.xlim(xmin=0, xmax=xy_max)
# plt.ylim(ymin=0, ymax=xy_max)
# 画像の幅と高さの最大値を取得する
max_width = max(df['Image_Width'])
max_height = max(df['Image_Height'])

# 横軸と縦軸の範囲を設定する
plt.xlim(xmin=0, xmax=max_width)  
plt.ylim(ymin=0, ymax=max_height)  

plt.ylabel('height', fontsize=25)
plt.xlabel('width', fontsize=25)

plt.savefig('Image_Size_Distribution.pdf', dpi=120, bbox_inches='tight')

plt.show()