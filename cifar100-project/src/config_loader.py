import yaml
import os

def load_config(path=None):
    if path is None:
        # Trouve config.yaml peu importe d'où on lance le script
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "..", "config", "config.yaml")
    
    with open(os.path.normpath(path), "r") as f:
        return yaml.safe_load(f)

# Usage dans n'importe quel fichier :
# from src.config_loader import load_config
# cfg = load_config()
# file_path = cfg["paths"]["raw_train"]