from __init__ import *

class ItemClassificationModel(nn.Module):
    def __init__(self, backbone="vit_base", pretrained=True, device=DEVICE):
        super(ItemClassificationModel, self).__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.device = device
        self.model = self._create_model()
        
    def _create_backbone(self):
        if self.backbone == "vit_base":
            return torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
        if self.backbone == "resnet50":
            return torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        return torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
        
    def _create_model(self):
        _model = self._create_backbone()
        if _model is None: return None
        
        if self.pretrained == True:
            for param in _model.parameters():
                param.requires_grad = False
                
        in_features = 768
        
        if self.backbone == "vit_base": in_features = 768
        if self.backbone == "resnet50": in_features = 2048
        
        classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 13)
        )
        
        if self.backbone == "vit_base": _model.heads = classifier
        if self.backbone == "resnet50": _model.fc = classifier
        return _model.to(self.device)
    
    def get_model(self):
        return self.model
    
    def forward(self, x):
        return self.model(x)
    
    
class OutfitClassificationModel(nn.Module):
    def __init__(self, device=DEVICE, input_size=2560, hidden_size=64, dropout_rate=0.1):
        super(OutfitClassificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out.squeeze(1)
    
    def get_model(self):
        return self.model