import torch
import torchvision


def get_mobilenet_model(num_classes: int = 29):
    # model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.mobilenet.MobileNet_V3_Small_Weights)
    # model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)

    model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)

    # model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    # model.heads.head = torch.nn.Linear( model.heads.head.in_features, num_classes)

    return model
