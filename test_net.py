import torch
from torchsummary import summary
from nets.resnet50 import ResNet, Bottleneck
from config import Config

def main():
    # 実行デバイスの決定（GPUが使えるならCUDA、そうでなければCPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルのインスタンス化
    # config.py の NUM_CLASSES（現在は2）を自動的に反映
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # ネットワーク構成のサマリーを表示
    # ヘルメット画像（RGB 3チャンネル）に合わせて 224x224 で検証
    print(f"[*] 使用デバイス: {device}")
    summary(model, input_size=(3, 224, 224))

if __name__ == "__main__":
    main()