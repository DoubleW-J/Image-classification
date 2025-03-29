import os
import shutil
import random
import pandas as pd

def create_directory_structure(dataset_path, classes):
    """ トレーニングとテストのフォルダ構造を作成 """
    # train と val フォルダを作成
    os.mkdir(os.path.join(dataset_path, 'train'))
    os.mkdir(os.path.join(dataset_path, 'val'))

    # 各クラスごとに train と val フォルダ内にサブフォルダを作成
    for fruit in classes:
        os.mkdir(os.path.join(dataset_path, 'train', fruit))
        os.mkdir(os.path.join(dataset_path, 'val', fruit))

def split_dataset(dataset_path, classes, test_frac=0.2):
    """ データセットをトレーニングセットとテストセットに分割し、画像ファイルを移動 """
    random.seed(123)  # 再現性のためランダムシードを設定
    df = pd.DataFrame()

    print('{:^18} {:^18} {:^18}'.format('Category', 'Number of training data', 'Number of test data'))

    for one_class in classes:
        old_dir = os.path.join(dataset_path, one_class)
        
        # フォルダが存在しない場合はスキップ
        try:
            images_filename = os.listdir(old_dir)
        except FileNotFoundError:
            print(f"The folder {old_dir} does not exist, skipping this category.")
            continue

        # 画像ファイルをランダムにシャッフル
        random.shuffle(images_filename)

        # テストセットとトレーニングセットの分割
        testset_number = int(len(images_filename) * test_frac)
        testset_images = images_filename[:testset_number]  # テストセット
        trainset_images = images_filename[testset_number:]  # トレーニングセット

        # 画像を test フォルダに移動
        for image in testset_images:
            old_img_path = os.path.join(dataset_path, one_class, image)
            new_test_path = os.path.join(dataset_path, 'val', one_class, image)
            shutil.move(old_img_path, new_test_path)

        # 画像を train フォルダに移動
        for image in trainset_images:
            old_img_path = os.path.join(dataset_path, one_class, image)
            new_train_path = os.path.join(dataset_path, 'train', one_class, image)
            shutil.move(old_img_path, new_train_path)

        # 元のフォルダを削除
        assert len(os.listdir(old_dir)) == 0  # フォルダが空であることを確認
        shutil.rmtree(old_dir)

        # 各クラスのデータ数を表示
        print('{:^18} {:^18} {:^18}'.format(one_class, len(trainset_images), len(testset_images)))

        # DataFrame に新しい行を追加
        new_row = pd.DataFrame({'class': [one_class], 'trainset': [len(trainset_images)], 'testset': [len(testset_images)]})
        df = pd.concat([df, new_row], ignore_index=True)

    return df

def main():
    dataset_path = r'D:\Download\dataset\helmet_classification'
    dataset_name = dataset_path.split('_')[0]
    print(f'Dataset: {dataset_name}')

    classes = os.listdir(dataset_path)
    create_directory_structure(dataset_path, classes)

    # データセットを分割し、統計情報を取得
    df = split_dataset(dataset_path, classes)

    # データセットフォルダをリネーム
    shutil.move(dataset_path, f"{dataset_name}_split")

    # データセットの統計情報を CSV ファイルに保存
    df['total'] = df['trainset'] + df['testset']
    df.to_csv('Data_volume_statistics.csv', index=False)

if __name__ == "__main__":
    main()
