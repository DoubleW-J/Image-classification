import torch
from nets.resnet50 import ResNet,Bottleneck
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import cv2
import time
import config


PATH = config.model_path
prediction_dir = config.prediction_dir
Batch_Size = int(input('Enter the number of images to predict in each batch:'))
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=config.num_classes)
model.load_state_dict(torch.load(PATH))
model = model.cuda()
model.eval()
test_transform = transforms.Compose([
    transforms.Resize(256),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 根据数据集调整均值和标准差
])
test_dataset = datasets.ImageFolder(prediction_dir, transform=test_transform)

gen_test = DataLoader(dataset=test_dataset, batch_size=Batch_Size, shuffle=True)

while True:

    images, lables = next(iter(gen_test))
    img = torchvision.utils.make_grid(images, nrow=Batch_Size)
    img_array = img.numpy().transpose(1, 2, 0)

    start_time = time.time()
    outputs = model(images.cuda())
    _, id = torch.max(outputs.data, 1)
    end_time = time.time()

    print('Prediction time:', end_time-start_time)
    print('Predicted result:', id.data.cpu().numpy())

    cv2.imshow('img', img_array)
    cv2.waitKey(0)


