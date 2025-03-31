import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import os
from nets.resnet50 import Bottleneck, ResNet
from tqdm import tqdm
import config 

# 訓練データの前処理：ランダムクロップとリサイズ、水平反転、テンソル変換、正規化
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # ランダムにクロップして224x224にリサイズ
    transforms.RandomHorizontalFlip(),  # ランダムに水平反転
    transforms.ToTensor(),  # テンソルに変換
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 正規化（ImageNetの平均と標準偏差）
])

# テストデータの前処理：リサイズ、センタークロップ、テンソル変換、正規化
test_transform = transforms.Compose([
    transforms.Resize(256),  # 短辺を256にリサイズ
    transforms.CenterCrop(224),  # 中央で224x224にクロップ
    transforms.ToTensor(),  # テンソルに変換
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 正規化（ImageNetの平均と標準偏差）
])



# データセットのパス設定
dataset_dir = config.dataset_dir
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')

# データセットの読み込み
train_dataset = datasets.ImageFolder(train_path, train_transform)
test_dataset = datasets.ImageFolder(test_path, test_transform)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, softmaxloss, epoch, epoch_size, epoch_size_val, gen, gen_test, Epoch, cuda, best_accuracy):
    total_loss = 0
    val_loss = 0
    test_correct = 0

    # 訓練フェーズ
    with tqdm(total=epoch_size, desc='Epoch{}/{}'.format(epoch + 1, Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            images, targets = batch[0], batch[1]
            images = images.to(device)
            targets = targets.to(device)

            # ----------------------#
            #   勾配の初期化
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   フォワードパス
            # ----------------------#
            outputs = net(images)
            # ----------------------#
            #   損失の計算
            # ----------------------#
            loss = softmaxloss(outputs, targets)
            # ----------------------#
            #   バックプロパゲーション
            # ----------------------#
            loss.backward()
            optimizer.step()
            total_loss += loss

            pbar.set_postfix(**{'total_loss': float(total_loss / (iteration + 1)),
                                'lr': get_lr(optimizer)})
            pbar.update(1)


    # 検証フェーズ
    net.eval()
    print('\nテスト開始')
    with tqdm(total=epoch_size_val, desc='Epoch{}/{}'.format(epoch + 1, Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_test):
            images, targets = batch[0], batch[1]
            images = images.to(device)
            targets = targets.to(device)
            outputs = net(images)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == targets.data)
            pbar.set_postfix(**{'test AP': float(100 * test_correct / len(test_dataset))})
            pbar.update(1)
    
    # 現在のモデルの検証精度を計算
    current_accuracy = float(100 * test_correct / len(test_dataset))
    print(f"Epoch {epoch + 1} Validation Accuracy: {current_accuracy}%")



    # 現在のモデルの検証精度がより良ければ、モデルを保存
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        
        # もし既存の最良モデルが存在すれば削除
        if best_accuracy > 0.0:
            old_model_path = f'logs/best_model_{best_accuracy:.2f}.pth'
            if os.path.exists(old_model_path):
                os.remove(old_model_path)  # 旧モデルを削除
        
        # 新しい最良モデルの保存
        model_save_path = f'logs/best_model_{current_accuracy:.2f}.pth'  # 精度を保存時のモデル名に含める
        torch.save(net.state_dict(), model_save_path)
        print(f"新しい最良モデルが保存されました。精度: {current_accuracy}%")

    return best_accuracy

if __name__ == '__main__':


    # ----------------------------#
    #   Cudaを使用するか
    #   GPUがない場合はFalseに設定
    # ----------------------------#
    cuda = config.cuda
    # ----------------------------#
    #   事前学習モデルを使用するか
    # ----------------------------#
    pre_train = config.pre_train
    # ----------------------------#
    #   コサイン退火学習率を使用するか
    # ----------------------------#
    CosineLR = config.CosineLR

    # ----------------------------#
    #   ハイパーパラメータ設定
    #   lr：学習率
    #   Batch_size：バッチサイズ
    # ----------------------------#
    lr = config.lr
    Batch_size = config.Batch_size
    Init_Epoch = config.Init_Epoch
    Fin_Epoch = config.Fin_Epoch

    # モデル作成
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=config.num_classes, pretrained=pre_train)
    
    if pre_train:
        model_path = config.pretrained_model_path
        pretrained_dict = torch.load(model_path, weights_only=True)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'conv1' not in k and 'fc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # データのロード
    gen = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0)
    gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_size // 2, shuffle=True, num_workers=0)

    epoch_size = len(gen)
    epoch_size_val = len(gen_test)

    softmax_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if CosineLR:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-10)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    # 最良の精度を追跡
    best_accuracy = 0.0

    for epoch in range(Init_Epoch, Fin_Epoch):
        best_accuracy = fit_one_epoch(net=model, softmaxloss=softmax_loss, epoch=epoch, epoch_size=epoch_size,
                                      epoch_size_val=epoch_size_val, gen=gen, gen_test=gen_test, Epoch=Fin_Epoch,
                                      cuda=cuda, best_accuracy=best_accuracy)
        lr_scheduler.step()
  
