import math

import torch
from torch import nn


def cnn_config(config=None):
    """Resolve legacy defaults and reject incompatible classifier settings."""
    cfg = {'architecture': 'cnn', 'input_size': 64, 'normalization': 'none',
           'weights': None, 'optimizer': 'adam', 'weight_decay': 0.0,
           'min_delta': 0.0, **(config or {})}
    expected = {'cnn': (64, 'none', None), 'resnet18': (224, 'imagenet', 'IMAGENET1K_V1')}
    if cfg['architecture'] not in expected:
        raise ValueError(f"Unknown architecture: {cfg['architecture']}")
    if tuple(cfg[k] for k in ['input_size', 'normalization', 'weights']) != expected[cfg['architecture']]:
        raise ValueError('Invalid classifier input_size, normalization or weights')
    if cfg['optimizer'] not in {'adam', 'adamw'} or cfg['weight_decay'] < 0:
        raise ValueError('Invalid classifier optimizer or weight_decay')
    if not math.isfinite(cfg['min_delta']) or cfg['min_delta'] < 0:
        raise ValueError('Invalid classifier min_delta: expected a finite nonnegative value')
    return cfg


class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Linear(512, 43)

    def forward(self, images):
        images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear',
                                          align_corners=False, antialias=True)
        return self.backbone((images - self.mean) / self.std)


def make_cnn(config=None, pretrained=False):
    cfg = cnn_config(config)
    if cfg['architecture'] == 'resnet18':
        return ResNet18(pretrained)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, 43))


def image_tensor(image):
    return torch.from_numpy(image.transpose(2, 0, 1).copy())
