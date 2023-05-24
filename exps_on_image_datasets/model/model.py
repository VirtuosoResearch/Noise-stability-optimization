import torch
import dill

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Identity
import torchvision.models as models
from model.modeling_vit import VisionTransformer

class MnistModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_classes, n_layers=1):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.feature_extractor = []
        for i in range(n_layers):
            if i == 0:
                self.feature_extractor.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.feature_extractor.append(nn.Linear(hidden_dim, hidden_dim))
            self.feature_extractor.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*self.feature_extractor)
        self.pred_head = nn.Linear(hidden_dim, n_classes)

    def reset_parameters(self, *args):
        for i in range(self.n_layers):
            self.feature_extractor[2*i].reset_parameters()
        self.pred_head.reset_parameters()

    def forward(self, x):
        batch_num = x.size(0)
        x = x.view(batch_num, -1)
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        return F.log_softmax(x, dim=1)

class IdentityModule(nn.Module):
    r"""An identity module that outputs the input."""

    def __init__(self) -> None:
        super(IdentityModule, self).__init__()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        r"""Forward function.

        Args:
          x(Tensor): Input tensor.

        Returns:
          Tensor: Output of identity module which is the same with input.

        """

        return x

class AlexNet(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10):
        super(AlexNet, self).__init__()
        self.feature_extractor = models.alexnet(pretrained=pretrained)

        self.in_features = self.feature_extractor.classifier[6].in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.classifier[6] = IdentityModule()

    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        return F.log_softmax(x, dim=1)

class VGG19(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10):
        super(VGG19, self).__init__()
        self.feature_extractor = models.vgg19(pretrained=pretrained)

        self.in_features = self.feature_extractor.classifier[6].in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.classifier[6] = IdentityModule()

    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        return F.log_softmax(x, dim=1)

class ResNet101(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10):
        super(ResNet101, self).__init__()
        self.feature_extractor = models.resnet101(pretrained = pretrained)

        self.in_features = self.feature_extractor.fc.in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.fc = IdentityModule()
    
    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        return F.log_softmax(x, dim=1)

class ResNet18(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10):
        super(ResNet18, self).__init__()
        self.feature_extractor = models.resnet18(pretrained = pretrained)

        self.in_features = self.feature_extractor.fc.in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.fc = IdentityModule()
    
    def load_robust_model(self, state_dict_dir):
        checkpoint = torch.load(state_dict_dir, pickle_module=dill)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint["state_dict"]
        state_dict = {k[len('module.model.'):]:v for k,v in state_dict.items()}
        self.feature_extractor.load_state_dict(state_dict, strict=False)
    
    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x, return_softmax = True):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        if return_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class ResNet50(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10):
        super(ResNet50, self).__init__()
        self.feature_extractor = models.resnet50(pretrained = pretrained)

        self.in_features = self.feature_extractor.fc.in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.fc = IdentityModule()
    
    def load_robust_model(self, state_dict_dir):
        checkpoint = torch.load(state_dict_dir, pickle_module=dill)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint["state_dict"]
        state_dict = {k[len('module.model.'):]:v for k,v in state_dict.items()}
        self.feature_extractor.load_state_dict(state_dict, strict=False)

    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        return F.log_softmax(x, dim=1)
    
class ResNet34(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10, return_softmax = True):
        super(ResNet34, self).__init__()
        self.feature_extractor = models.resnet34(pretrained = pretrained)

        self.in_features = self.feature_extractor.fc.in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.fc = IdentityModule()
        self.return_softmax = return_softmax
    
    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        if self.return_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x
    
class ViTB16(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10, return_softmax = True):
        super(ViTB16, self).__init__()
        self.feature_extractor = models.vit_b_16(weights=models.ViT_B_16_Weights)

        self.in_features = self.feature_extractor.heads[0].in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.heads = IdentityModule()
        self.return_softmax = return_softmax
    
    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        if self.return_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x
        
class ViTL16(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10, return_softmax = True):
        super(ViTL16, self).__init__()
        self.feature_extractor = models.vit_l_16(weights=models.ViT_L_16_Weights)

        self.in_features = self.feature_extractor.heads[0].in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.heads = IdentityModule()
        self.return_softmax = return_softmax
    
    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        if self.return_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class EfficientNetB7(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10):
        super(EfficientNetB7, self).__init__()
        self.feature_extractor = models.efficientnet_b7(pretrained = pretrained)

        self.in_features = self.feature_extractor.classifier[1].in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.classifier[1] = IdentityModule()
    
    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        return F.log_softmax(x, dim=1)
        
class EfficientNetL2(nn.Module):

    def __init__(self, pretrained = True, n_classes = 10):
        super(EfficientNetL2, self).__init__()
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
        self.feature_extractor = models.efficientnet_v2_l(weights=weights)

        self.in_features = self.feature_extractor.classifier[1].in_features
        self.out_features = n_classes
        self.pred_head = nn.Linear(self.in_features, self.out_features)
        self.feature_extractor.classifier[1] = IdentityModule()
    
    def reset_parameters(self, state_dict = None):
        # Reload source dict
        if state_dict is not None: 
            self.load_state_dict(state_dict)
        self.pred_head.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pred_head(x)
        return F.log_softmax(x, dim=1)
        
