import torch
from torchsummary import summary
import config


from nets.resnet50 import ResNet,Bottleneck
from torch import sum
if __name__ == "__main__":
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=config.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(3, 28, 28))

pass