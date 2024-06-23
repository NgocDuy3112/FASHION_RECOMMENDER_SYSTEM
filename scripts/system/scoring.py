from models import *
from helper import *

class ScoringModule():
    def __init__(self, checkpoint_path=OUTFIT_CHECKPOINT_PATH, device=DEVICE):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = self._create_model()
        self.model.eval()
    
    def _create_model(self):
        model = OutfitClassificationModel(self.device).get_model()
        if self.checkpoint_path:
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device(self.device)), strict=False)
        return model
    
    def score(self, embeddings_list):
        idx = 0
        vector = np.zeros(shape=2560, dtype=np.float32)
        for embedding in embeddings_list:
            vector[idx: idx + 512] = embedding.copy()
            idx += 512
        vector = torch.from_numpy(vector).to(self.device)
        with torch.no_grad():
            outputs = F.sigmoid(self.model(vector))
        return outputs