import torch
from nets.resnet50 import ResNet,Bottleneck
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import config

Batch_size = config.Batch_size
test_path = config.test_path
root = '.\logs'
file_dir = os.listdir(root)
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=config.num_classes)

for file in file_dir:
    PATH = os.path.join(root, file)



    model.load_state_dict(torch.load(PATH))
    model = model.cuda()
    model.eval()

    # データの前処理：必要に応じて調整してください
    test_transform = transforms.Compose([
        transforms.Resize(256),  # 必要に応じて調整できます
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # データセットに合わせて平均と標準偏差を調整してください
    ])

    # カスタムテストセットの読み込み（ImageFolderを使用）
    test_dataset = datasets.ImageFolder(test_dataset, transform=test_transform)


    gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=True)

    test_correct = 0
    for data in gen_test:
        inputs, lables = data
        inputs, lables = Variable(inputs).cuda(), Variable(lables).cuda()
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == lables.data)
    print(file)
    print("correct:%.3f%%" % (100 * test_correct / len(test_dataset)))
