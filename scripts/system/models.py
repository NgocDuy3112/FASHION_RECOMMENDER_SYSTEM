from __init__ import *

class ItemClassificationModel(nn.Module):
    def __init__(self, backbone="vit_base", pretrained=True):
        super(ItemClassificationModel, self).__init__()
        self.backbone = backbone
        self.pretrained = pretrained
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
        return _model
    
    def get_model(self):
        return self.model
    
    def forward(self, x):
        return self.model(x)    
    
    
class OutfitClassificationMLPModel(nn.Module):
    def __init__(self, input_size=2560, hidden_size=64, dropout_rate=0.1):
        super(OutfitClassificationMLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        out = self.model(x)
        # return out.squeeze(1)
        return out
    
    def get_model(self):
        return self.model
    
    
class OutfitClassificationBiLSTMModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, num_classes=1, device="cpu"):
        super(OutfitClassificationBiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.device = device
    
    def forward(self, x, mask):
        # Apply mask to ignore zeros
        x = x * mask.unsqueeze(2)
        
        # Get lengths of non-padded sequences and move to CPU
        lengths = mask.sum(dim=1).int().cpu()
        
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (_, _) = self.bilstm(packed_input)
        
        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Take the last hidden state for classification
        last_hidden_states = lstm_out[torch.arange(lstm_out.size(0)), lengths - 1, :].to(self.device)
        
        out = self.fc(last_hidden_states)
        return out
    
    def get_model(self):
        return self.model
