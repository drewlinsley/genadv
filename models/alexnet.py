import torch
import torch.nn as nn
from models.resnet_utils import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, siamese=False, version='subtract'):
        super(AlexNet, self).__init__()
        self.siamese = siamese
        self.version = version
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        fan_in = 256 * 6 * 6
        if self.siamese and version == 'cat':
            fan_in *= 2
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fan_in, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        if self.siamese:
            x0 = x[..., 0]
            x1 = x[..., 1]
            x0 = self.features(x0)
            x0 = self.avgpool(x0)
            x0 = torch.flatten(x0, 1)
            x1 = self.features(x1)
            x1 = self.avgpool(x1)
            x1 = torch.flatten(x1, 1)
            if self.version == 'cat':
                x = self.classifier(torch.stack((x0, x1), -1))
            else:
                x = self.classifier(torch.abs(x0 - x1))
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x

    def _initialize(self):
        if self.pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                                  progress=False)
            self.load_state_dict(state_dict)
        else:
            raise NotImplementedError('Need routine for init from scratch.')


def alexnet(siamese, version, pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(siamese=siamese, version=version, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

