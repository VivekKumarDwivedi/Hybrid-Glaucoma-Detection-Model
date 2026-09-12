import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

from src.dataset import GlaucomaDataset, get_transforms
from src.models import get_model
from src.utils import (
    plot_fold_summary,
    plot_roc_and_cm,
    plot_probability_distribution,
    plot_gradcam,
    plot_architecture_summary,
)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    preds, probs, truths = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            prob = torch.softmax(outputs, dim=1)[:, 1]
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            
            preds.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())
            truths.extend(labels.cpu().numpy())
            
    return total_loss / len(loader), correct / len(loader.dataset), np.array(preds), np.array(probs), np.array(truths)

def run_cross_validation(images, labels, names, config):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=config['training']['n_folds'], shuffle=True, random_state=config['project']['seed'])

    all_preds, all_probs, all_labels = [], [], []
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(images, labels_arr), 1):
        print(f"\n--- Fold {fold}/{config['training']['n_folds']} ---")
        train_imgs, train_lbls = [images[i] for i in tr_idx], labels_arr[tr_idx].tolist()
        val_imgs, val_lbls = [images[i] for i in va_idx], labels_arr[va_idx].tolist()
        val_names = [names[i] for i in va_idx]

        counts = Counter(train_lbls)
        weights = [1.0 / counts[l] for l in train_lbls]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        tr_ds = GlaucomaDataset(train_imgs, train_lbls, get_transforms(config['data']['img_size'], is_train=True))
        va_ds = GlaucomaDataset(val_imgs, val_lbls, get_transforms(config['data']['img_size'], is_train=False))

        tr_loader = DataLoader(tr_ds, batch_size=config['training']['batch_size'], sampler=sampler)
        va_loader = DataLoader(va_ds, batch_size=config['training']['batch_size'], shuffle=False)

        model = get_model(config['training']['model_name'], config['data']['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

        best_auc = 0
        best_state = None

        for epoch in range(1, config['training']['epochs'] + 1):
            tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, device)
            va_loss, va_acc, preds, probs, truths = eval_epoch(model, va_loader, criterion, device)
            
            fold_auc = roc_auc_score(truths, probs) if len(np.unique(truths)) > 1 else 0.5
            if fold_auc >= best_auc:
                best_auc = fold_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                print(f"Ep {epoch:02d} | Train Acc: {tr_acc:.3f} | Val Acc: {va_acc:.3f} | Val AUC: {fold_auc:.3f}")

        model.load_state_dict(best_state)
        _, _, preds, probs, truths = eval_epoch(model, va_loader, criterion, device)
        all_preds.extend(preds); all_probs.extend(probs); all_labels.extend(truths)

        fold_results.append({
            'fold': fold,
            'acc': accuracy_score(truths, preds),
            'f1': f1_score(truths, preds, zero_division=0),
            'auc': roc_auc_score(truths, probs) if len(np.unique(truths)) > 1 else 0.5,
        })

    output_dir = config['project']['output_dir']
    plot_fold_summary(fold_results, output_dir=output_dir)
    plot_roc_and_cm(
        np.asarray(all_labels),
        np.asarray(all_probs),
        np.asarray(all_preds),
        output_dir=output_dir,
    )
    plot_probability_distribution(
        np.asarray(all_labels),
        np.asarray(all_probs),
        output_dir=output_dir,
    )
    plot_gradcam(
        model,
        val_imgs,
        val_lbls,
        val_names,
        get_transforms,
        device,
        img_size=config['data']['img_size'],
        output_dir=output_dir,
    )

    print("\n" + "="*50)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Glaucoma']))

    os.makedirs(f"{config['project']['output_dir']}/checkpoints", exist_ok=True)
    save_path = f"{config['project']['output_dir']}/checkpoints/best_{config['training']['model_name']}_model.pth"
    torch.save(best_state, save_path)
    print(f"Saved best model checkpoint to {save_path}")
    plot_architecture_summary(model)