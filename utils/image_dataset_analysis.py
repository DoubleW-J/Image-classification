import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 1️⃣ 画像データを読み込む関数
def load_dataset(dataset_path):
    data = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    for fruit in tqdm(os.listdir(dataset_path)):  # フォルダ内のカテゴリ（果物の種類）を繰り返し処理
        fruit_path = os.path.join(dataset_path, fruit)
        
        if not os.path.isdir(fruit_path):  
            continue  # フォルダでない場合はスキップ
        
        for file in os.listdir(fruit_path):  # そのカテゴリ内のすべての画像を処理
            if not any(file.lower().endswith(ext) for ext in valid_extensions):
                continue  # 画像ファイルでない場合はスキップ
            
            img_path = os.path.join(fruit_path, file)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Error reading image: {img_path}")
                continue
            
            data.append({
                'Category': fruit,
                'Filename': file,
                'Image_Width': img.shape[1],
                'Image_Height': img.shape[0]
            })
    
    return pd.DataFrame(data)

# 2️⃣ 画像サイズ分布をプロットする関数
def plot_image_distribution(df):
    x = df['Image_Width']
    y = df['Image_Height']
    
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # カーネル密度推定（KDE）を使用して密度を計算
    
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]  # 密度の低い順に並び替え

    plt.figure(figsize=(10,10))
    plt.scatter(x, y, c=z, s=5, cmap='Spectral_r')  # 散布図を描画
    plt.tick_params(labelsize=15)

    max_width = max(df['Image_Width'])
    max_height = max(df['Image_Height'])
    
    plt.xlim(xmin=0, xmax=max_width)  # X軸の範囲を設定
    plt.ylim(ymin=0, ymax=max_height)  # Y軸の範囲を設定

    plt.ylabel('Height', fontsize=25)
    plt.xlabel('Width', fontsize=25)
    
    plt.savefig('Image_Size_Distribution.pdf', dpi=120, bbox_inches='tight')  # PDFに保存
    plt.show()

# 3️⃣ メイン関数
def main():
    dataset_path = r"D:\Download\dataset\helmet_classification" 
    
    # データを読み込む
    df = load_dataset(dataset_path)
    
    # CSVファイルに保存する
    df.to_csv('helmet_classification.csv', index=False)
    print("Data has been saved to 'helmet_classification.csv'") 
    
    # 画像サイズ分布を可視化する
    plot_image_distribution(df)

# 4️⃣ エントリーポイント
if __name__ == "__main__":
    main()
