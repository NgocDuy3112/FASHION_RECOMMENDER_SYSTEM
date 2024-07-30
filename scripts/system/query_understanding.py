from models import *

class QueryUnderstandingModule():
    def __init__(
        self, 
        checkpoint_path=ITEMS_CHECKPOINT_PATH, 
        backbone="vit_base", 
        pretrained=True, 
        device=DEVICE, 
        embedding_size=512
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.pretrained = pretrained
        self.embedding_size = embedding_size
        
        self.model = self._create_model()
        self.node = self._get_node()
        self.feature_extractor = self._create_feature_extractor()
    
    def _create_model(self):
        model = ItemClassificationModel(self.backbone, self.pretrained).get_model()
        if self.checkpoint_path:
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device(self.device)), strict=False)
        return model
    
    def _get_node(self):
        if self.backbone == "vit_base":
            return {'heads.6': 'heads.6'} if self.embedding_size == 256 else {'heads.3': 'heads.3'} if self.embedding_size == 512 else None
        if self.backbone == "resnet50":
            return {'fc.6': 'fc.6'} if self.embedding_size == 256 else {'fc.3': 'fc.3'} if self.embedding_size == 512 else None
        return None
    
    def _get_node_str(self):
        if self.backbone == "vit_base":
            return 'heads.6' if self.embedding_size == 256 else 'heads.3' if self.embedding_size == 512 else None
        if self.backbone == "resnet50":
            return 'fc.6' if self.embedding_size == 256 else 'fc.3' if self.embedding_size == 512 else None
        return None
    
    def get_model(self):
        return self.model
    
    def _create_feature_extractor(self):
        if self.node is None: return None
        return create_feature_extractor(self.model, self.node)
        
    def _get_category(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            label = ITEM_LABEL_DICT[preds[0]]
        return label
    
    def _get_embeddings(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.feature_extractor(x)
            embeddings = outputs[self._get_node_str()].cpu().numpy()
        return embeddings.reshape(-1, )
    
    def forward(self, x):
        return self._get_embeddings(x), self._get_category(x)
    
    
if __name__ == "__main__":
    query_understanding_module = QueryUnderstandingModule(device="cpu")
    print(query_understanding_module.forward(torch.rand(1, 3, 224, 224)))
