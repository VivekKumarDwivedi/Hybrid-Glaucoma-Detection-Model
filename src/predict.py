import argparse
import os
import yaml
import torch
import cv2
import numpy as np
from PIL import Image

from src.models import get_model
from src.dataset import get_transforms
from src.utils import GradCAM

def predict_single_image(image_path, model_path, config_path="config/config.yaml", save_heatmap=True):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # 2. Load Model Architecture & Checkpoint
    model = get_model(config['training']['model_name'], config['data']['num_classes']).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both full checkpoint dicts and state dicts
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load & Preprocess Image
    raw_img = Image.open(image_path).convert("RGB")
    transform = get_transforms(config['data']['img_size'], is_train=False)
    tensor_img = transform(raw_img).unsqueeze(0).to(device)

    # 4. Model Prediction
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1)[0]
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probs[predicted_class].item()

    class_names = ["Normal", "Glaucoma Suspect"]
    result = {
        "prediction": class_names[predicted_class],
        "glaucoma_probability": round(probs[1].item() * 100, 2),
        "confidence": round(confidence * 100, 2)
    }

    print(f"\n--- Prediction Output ---")
    print(f"File       : {image_path}")
    print(f"Result     : {result['prediction']}")
    print(f"Glaucoma % : {result['glaucoma_probability']}%")

    # 5. Optional Grad-CAM Heatmap Generation
    if save_heatmap:
        if config['training']['model_name'] == 'ensemble':
            target_layers = [model.resnet.layer4[-1], model.densenet.features[-1]]
        else:
            target_layer = model.layer4[-1] if hasattr(model, 'layer4') else model.features[-1]
            target_layers = [target_layer]

        heatmaps = [
            GradCAM(model, target_layer).generate_heatmap(tensor_img, target_class=1)
            for target_layer in target_layers
        ]
        heatmap = np.mean(heatmaps, axis=0)

        # Overlay Heatmap on Original Image
        orig = np.array(raw_img.resize((config['data']['img_size'], config['data']['img_size'])))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)

        out_heatmap_path = "outputs/plots/latest_prediction_heatmap.png"
        os.makedirs(os.path.dirname(out_heatmap_path), exist_ok=True)
        cv2.imwrite(out_heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM Visual saved to: {out_heatmap_path}")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glaucoma Image Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input OCT or Fundus image")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/best_densenet_model.pth", help="Path to checkpoint")
    args = parser.parse_args()

    predict_single_image(args.image, args.model)