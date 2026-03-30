from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ViT_B_16_Weights,
    EfficientNet_V2_S_Weights,
)


# =========================
# ConvNeXt
# =========================
def get_convnext_valid_transforms():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    return weights.transforms()


def get_convnext_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3),
            value="random"
        ),
    ])


def get_convnext256_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=256,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3),
            value="random"
        ),
    ])


def get_convnext256_valid_transforms():
    return transforms.Compose([
        transforms.Resize(276),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =========================
# EfficientNetV2 S
# =========================
def get_efficientnetv2s_valid_transforms():
    weights = EfficientNet_V2_S_Weights.DEFAULT
    return weights.transforms()


def get_efficientnetv2s_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3),
            value="random"
        ),
    ])


# =========================
# ViT
# =========================
def get_vit_valid_transforms():
    weights = ViT_B_16_Weights.DEFAULT
    return weights.transforms()


def get_vit_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])