import torch
import torch.nn as nn
from torchvision import models

class GlaucomaEnsemble(nn.Module):
    """
    Ensemble model combining ResNet-50 and DenseNet-121.
    Averages prediction logits from both backbone networks.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Backbone 1: ResNet-50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        res_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(res_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Backbone 2: DenseNet-121
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        dense_in = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(dense_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out_resnet = self.resnet(x)
        out_densenet = self.densenet(x)
        return (out_resnet + out_densenet) / 2.0


def _build_resnet(variant='resnet50', num_classes=2):
    """Helper function to build and adapt ResNet variants."""
    weights_dict = {
        'resnet18': models.ResNet18_Weights.DEFAULT,
        'resnet34': models.ResNet34_Weights.DEFAULT,
        'resnet50': models.ResNet50_Weights.DEFAULT,
    }
    model_fn_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }

    model = model_fn_dict[variant](weights=weights_dict[variant])
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model


def get_model(model_name='densenet', num_classes=2):
    """
    Unified Factory Function to retrieve the requested model architecture.
    """
    name = model_name.lower().strip()

    if name in ('resnet18', 'resnet34', 'resnet50', 'resnet'):
        variant = 'resnet50' if name == 'resnet' else name
        return _build_resnet(variant=variant, num_classes=num_classes)

    elif name == 'densenet':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        return model

    elif name == 'ensemble':
        return GlaucomaEnsemble(num_classes=num_classes)

    else:
        valid_models = ['resnet18', 'resnet34', 'resnet50', 'densenet', 'ensemble']
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from: {valid_models}")