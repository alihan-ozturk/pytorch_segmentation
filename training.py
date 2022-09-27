import gc
import os
import sys

import torch
import warnings
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("nextLife")

gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name('cuda')
print(device_name)

warnings.filterwarnings('ignore')

IMAGE_DIR = r"C:\Users\RoboGor\Desktop\IMG"
MASK_DIR = r"C:\Users\RoboGor\Desktop\MASK"

LEARNING_RATE = 0.0005
EPOCHS = 100
BATCH_SIZE = 10

augmentations = A.Compose([
    A.OneOrOther(A.RandomCrop(width=800, height=800, p=0.5), A.RandomResizedCrop(width=800, height=800, p=0.5)),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(),
    A.OneOrOther(A.MotionBlur(p=0.3), A.Blur(p=0.3)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Resize(320, 320),
    ToTensorV2()
])


class aerialDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_names = os.listdir(img_path)
        self.mask_names = os.listdir(mask_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_names[idx])
        mask_path = os.path.join(self.mask_path, self.mask_names[idx])
        image = np.array(Image.open(img_path))
        _, _, d = image.shape
        if d == 4:
            image = image[:, :, 0:-1]

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            transform = self.transform(image=image, mask=mask)
            image = transform["image"]
            mask = transform["mask"]

        image = image / 255.0

        return image, mask


data = aerialDataset(IMAGE_DIR, MASK_DIR, transform=augmentations)

data_size = len(data)
train_size = int(data_size * 0.8)
train_data, test_data = random_split(data, [train_size, data_size - train_size])
print(len(train_data), len(test_data))

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class ConvBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, bias=False, reluInplace=True):
        super(ConvBatchNormReLU, self).__init__()
        self.seqLayers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=reluInplace)
        )

    def forward(self, x):
        return self.seqLayers(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, poolKernel=2, poolStride=2, copySizes=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downConv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=poolKernel, stride=poolStride)
        self.upsampling = nn.Upsample(scale_factor=(poolKernel, poolKernel), mode='bilinear')
        self.conv = nn.ModuleList()
        self.upConv = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        for numberOfFeatures in copySizes:
            self.downConv.append(nn.Sequential(
                ConvBatchNormReLU(in_channels, numberOfFeatures),
                ConvBatchNormReLU(numberOfFeatures, numberOfFeatures)
            ))
            in_channels = numberOfFeatures

        self.bottleneck = nn.Sequential(
            ConvBatchNormReLU(copySizes[-1], copySizes[-1] * 2),
            ConvBatchNormReLU(copySizes[-1] * 2, copySizes[-1] * 2)
        )

        for numberOfFeatures in reversed(copySizes):
            self.upConv.append(nn.Sequential(
                ConvBatchNormReLU(numberOfFeatures * poolKernel, numberOfFeatures),
                ConvBatchNormReLU(numberOfFeatures, numberOfFeatures)
            ))

            self.conv.append(nn.Sequential(
                nn.Conv2d(numberOfFeatures * 2, numberOfFeatures, kernel_size=3, stride=1, padding=1, bias=False)
            ))

        self.convFinal = nn.Conv2d(copySizes[0], out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        copies = []
        for doubleConv in self.downConv:
            x = doubleConv(x)
            copies.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        copies = copies[::-1]

        for index in range(len(self.upConv)):
            x = self.upsampling(x)
            x = self.conv[index](x)
            if x.shape != copies[index].shape:
                _, _, height, width = x.shape
                croppedCopy = TF.resize(copies[index], size=(height, width))
                x = torch.cat((croppedCopy, x), dim=1)
            else:
                x = torch.cat((copies[index], x), dim=1)

            x = self.upConv[index](x)

        x = self.convFinal(x)
        x = self.sigmoid(x)

        return x


model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("mart2.pth"))

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def acc(output, target):
    with torch.no_grad():
        predicted = torch.round(output)
        eq = torch.eq(predicted, target).int()
        return eq.sum() / eq.numel()


for epoch in range(EPOCHS):
    model.train()
    loss_train = 0
    loss_test = 0

    for i, (images, masks) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(images.to(device))
        masks = masks.unsqueeze(1).to(device)
        loss = criterion(outputs, masks)
        loss_train += loss.item()
        loss.backward()
        optimizer.step()

    accuracy = acc(outputs, masks)
    writer.add_scalar("Train Loss", loss_train / len(train_dataloader), epoch)
    writer.add_scalar("Train Acc", accuracy, epoch)
    print("TRAIN {}.epoch, loss : {}, acc : {}".format(epoch + 1, loss_train / len(train_dataloader), accuracy))

    model.eval()
    for j, (images, masks) in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(images.to(device))

        masks = masks.unsqueeze(1).to(device)
        loss = criterion(outputs, masks)
        loss_test += loss.item()

    accuracy = acc(outputs, masks)
    writer.add_scalar("Test Loss", loss_test / len(test_dataloader), epoch)
    writer.add_scalar("Test Acc", accuracy, epoch)
    print("TEST {}.epoch, loss : {}, acc : {}".format(epoch + 1, loss_test / len(test_dataloader), accuracy))

torch.save(model.state_dict(), "nextLife.pth")

# model.load_state_dict(torch.load("mart2.pth"))
model.eval()

for i, (images, masks) in enumerate(train_dataloader):
    with torch.no_grad():
        outputs = model(images.to(device))

    masks = masks.unsqueeze(1).to(device)
    loss = criterion(outputs, masks)
    outputs[outputs >= 0.5] = 1.0
    outputs[outputs < 0.5] = 0.0
    print("acc : {}".format(acc(outputs, masks)))

    for j in range(len(images)):

        image = images[j] * 255
        out = outputs[j] * 255
        mask = masks[j] * 255
        image = torch.permute(image, (1, 2, 0))
        out = torch.permute(out, (1, 2, 0))
        mask = torch.permute(mask, (1, 2, 0))

        image = image.cpu().numpy().astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        predict = out.cpu().numpy().astype('uint8')
        mask = mask.cpu().numpy().astype('uint8')
        cv2.imshow('original image', image)
        cv2.imshow('output', predict)
        cv2.imshow('mask', mask)

        key = cv2.waitKey(0)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
