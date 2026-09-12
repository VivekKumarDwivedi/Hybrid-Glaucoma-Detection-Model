import yaml
import random
import numpy as np
import torch
from src.dataset import load_data_from_zip
from src.train import run_cross_validation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['project']['seed'])
    print(f"Running pipeline with configuration: {config['training']['model_name'].upper()}")

    images, labels, names = load_data_from_zip(config['data']['file_path'])
    print(f"Dataset successfully loaded: {len(images)} images found.")

    run_cross_validation(images, labels, names, config)

if __name__ == "__main__":
    main()