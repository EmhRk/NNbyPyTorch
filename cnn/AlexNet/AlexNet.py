import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# AlexNet train with 2 kernel, but only one kernel will be used in thus code
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),

            nn.Conv2d(256, 384, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(364),

            nn.Conv2d(384, 384, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(384, 256, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),

            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(),

            nn.Linear(4096, num_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
