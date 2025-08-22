from pytorch_device_selection import select_torch_device

class Config:
    """Classe de configuration globale"""
    
    # Chemins
    DATA_DIR = "./data"
    RAW_DATA = f"{DATA_DIR}/raw"
    PROCESSED_DATA = f"{DATA_DIR}/processed"
    
    # Hyperparamètres
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 256
    
    # Méthode pour mettre à jour les paramètres
    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

# Instance globale
config = Config()