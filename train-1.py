import time
import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, accuracy_score, 
                             f1_score, roc_auc_score)
import wandb
import config
from tqdm import tqdm


# GPU があれば GPU を使用し、なければ CPU を使用する
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# 画像の前処理：スケーリング・クロッピング、データ拡張、Tensor 変換、正規化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

# テスト用画像の前処理-RCTN：スケーリング、クロッピング、Tensor 変換、正規化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])

# データセットのパスを指定する
dataset_dir = config.dataset_dir

train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')

# 訓練データの読み込み
train_dataset = datasets.ImageFolder(train_path, train_transform)
# テストデータの読み込み
test_dataset = datasets.ImageFolder(test_path, test_transform)

# 各クラスの名前
class_names = train_dataset.classes
n_class = len(class_names)
# マッピング関係：クラス → インデックス番号
train_dataset.class_to_idx
# マッピング関係：インデックス番号 → クラス
idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
# py ファイルとして保存
np.save('idx_to_labels.npy', idx_to_labels)
np.save('labels_to_idx.npy', train_dataset.class_to_idx)


# BATCH_SIZEを指定する
BATCH_SIZE = config.BATCH_SIZE


train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4
                         )

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4
                        )

# ImageNet データセットで事前学習された ResNetモデルをロードできます

# 最後の全結合層のパラメータのみを微調整し、他の層は凍結する

# 事前学習済みのモデルをロード
# model = models.resnet18(pretrained=True)

# # 最後の全結合層を変更し、現在のデータセットのクラス数に合わせる
# # 新しく作成した層はデフォルトで requires_grad=True
# model.fc = nn.Linear(model.fc.in_features, n_class)
# print(model.fc)
# optimizer = optim.Adam(model.fc.parameters())

# モデルの全てのパラメータを微調整

# 事前学習済みのモデルをロード
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, n_class)
# すべての層のパラメータを更新可能にする
optimizer = optim.Adam(model.parameters())

# 事前学習済みの重みを使わずに、モデルの重みをランダムに初期化する
# model = models.resnet18(pretrained=False) 
# model.fc = nn.Linear(model.fc.in_features, n_class)
# optimizer = optim.Adam(model.parameters())

# モデルを指定したデバイス（GPU または CPU）に移動
model = model.to(device)

# クロスエントロピー損失関数
criterion = nn.CrossEntropyLoss()

# 訓練エポック数
EPOCHS = config.EPOCHS

# 学習率減衰の戦略
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_one_batch(images, labels):
    '''
    1 バッチのトレーニングを実行し、現在のバッチのトレーニングログを返す
    '''
    
    # 1 バッチ分のデータとラベルをデバイスに転送
    images = images.to(device)
    labels = labels.to(device)
    # 前向き伝播
    outputs = model(images)
    # 損失を計算
    loss = criterion(outputs, labels)  # 現在のバッチでの平均クロスエントロピー損失を計算
    # 勾配の初期化
    optimizer.zero_grad()
    # 逆伝播
    loss.backward()
    optimizer.step()
    
    # 現在のバッチのラベルと予測されたラベルを取得
    _, preds = torch.max(outputs, 1)  # 現在のバッチのすべての画像の予測クラスを取得
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    # 分類の評価指標を計算
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    # log_train['train_precision'] = precision_score(labels, preds, average='macro')
    # log_train['train_recall'] = recall_score(labels, preds, average='macro')
    # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')
    
    return log_train


def evaluate_testset():
    '''
    テストセット全体で評価を行い、分類の評価指標ログを返す
    '''

    loss_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:  # 1 バッチ分のデータとラベルを取得
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # モデルに入力し、前向き予測を実行

            # テストセット全体のラベルと予測されたラベルを取得
            _, preds = torch.max(outputs, 1)  # 現在のバッチのすべての画像の予測クラスを取得
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # ロジットから現在のバッチの平均交差エントロピー損失を計算
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        
    log_test = {}
    log_test['epoch'] = epoch
    
    # 分類の評価指標を計算
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    
    return log_test


if __name__ == '__main__':
    epoch = 0
    batch_idx = 0
    best_test_accuracy = 0


    # 訓練ログ - 訓練データセット
    df_train_log = pd.DataFrame()
    log_train = {}
    log_train['epoch'] = 0
    log_train['batch'] = 0
    images, labels = next(iter(train_loader))
    log_train.update(train_one_batch(images, labels))
    df_train_log = df_train_log._append(log_train, ignore_index=True)

    # 訓練ログ - テストデータセット
    df_test_log = pd.DataFrame()
    log_test = {}
    log_test['epoch'] = 0
    log_test.update(evaluate_testset())
    df_test_log = df_test_log._append(log_test, ignore_index=True)
   
    wandb.init(project='fruit30', name=time.strftime('%m%d%H%M%S'))

    for epoch in range(1, EPOCHS+1):
    
        print(f'Epoch {epoch}/{EPOCHS}')

        # 訓練フェーズ
        model.train()
        for images, labels in tqdm(train_loader): # 1バッチのデータとラベルを取得
            batch_idx += 1
            log_train = train_one_batch(images, labels)
            df_train_log = df_train_log._append(log_train, ignore_index=True)
            wandb.log(log_train)
        
        lr_scheduler.step()

        # テストフェーズ
        model.eval()
        log_test = evaluate_testset()
        df_test_log = df_test_log._append(log_test, ignore_index=True)
        wandb.log(log_test)
    
        # 最新の最良モデルファイルを保存
        if log_test['test_accuracy'] > best_test_accuracy: 
            # 古い最良モデルファイルを削除（存在する場合）
            old_best_checkpoint_path = config.BEST_MODEL_PATH.format(best_test_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 新しい最良モデルファイルを保存
            best_test_accuracy = log_test['test_accuracy']
            new_best_checkpoint_path =  config.BEST_MODEL_PATH.format(log_test['test_accuracy'])
            torch.save(model, new_best_checkpoint_path)
            print(f'Save the new best model {config.BEST_MODEL_PATH.format(best_test_accuracy)}')
            # best_test_accuracy = log_test['test_accuracy']

    df_train_log.to_csv('training_log_train_data.csv', index=False)
    df_test_log.to_csv('training_log_test_data.csv', index=False)