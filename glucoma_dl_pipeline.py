"""
=============================================================
  Glaucoma Detection — Deep Learning Pipeline
  Dataset : OCT & Fundus Images (26 patients, 50 fundus imgs)
  Model   : Custom Deep CNN with Attention, Grad-CAM, Ensemble
=============================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os, re, zipfile, warnings, random
from io import BytesIO
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score, f1_score)

warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────────
FILE_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'Data on OCT and Fundus Images')
OUT_DIR    = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SEED       = 42
IMG_SIZE   = 128
BATCH_SIZE = 8
EPOCHS     = 40
LR         = 3e-4
N_FOLDS    = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_labels(folder_path):
    xlsx_files = []
    for root, _, files in os.walk(folder_path):
        xlsx_files.extend(os.path.join(root, name)
                          for name in files if name.lower().endswith('.xlsx'))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx label file found in: {folder_path}")

    df = pd.read_excel(xlsx_files[0], header=None)

    data = df.iloc[1:].copy()
    data.columns = range(data.shape[1])
    records = []
    for _, row in data.iterrows():
        name = str(row[0]).strip().strip("'").strip()
        if not name or name == 'nan':
            continue
        cdrs, labels = [], []
        for c in [1, 3, 5, 7]:
            try:
                cdrs.append(float(row[c]))
            except:
                pass
            lab = str(row[c+1]).strip().lower() if pd.notna(row[c+1]) else ''
            if lab in ('yes', 'no', 'suspect'):
                labels.append(lab)
        if not labels:
            continue
        vote   = Counter(labels).most_common(1)[0][0]
        binary = 1 if vote in ('yes', 'suspect') else 0
        if not name.lower().endswith('.jpg'):
            name += '.jpg'
        records.append({'image_name': name, 'label': binary,
                        'mean_cdr': np.mean(cdrs) if cdrs else np.nan})
    return pd.DataFrame(records)


def load_images(folder_path, label_df):
    images, labels, names = [], [], []
    image_map = {}
    for root, _, files in os.walk(folder_path):
        for name in files:
            if name.lower().endswith('.jpg'):
                image_map[name] = os.path.join(root, name)

    def open_image(path):
        return Image.open(path).convert('RGB')

    for _, row in label_df.iterrows():
        key = os.path.basename(row['image_name'])
        match = image_map.get(key)
        if not match:
            clean = re.sub(r"['\s]", "", key)
            for image_name, image_path in image_map.items():
                if re.sub(r"['\s]", "", image_name) == clean:
                    match = image_path
                    break
        if match:
            images.append(open_image(match))
            labels.append(row['label'])
            names.append(key)

    print(f"Loaded {len(images)}/{len(label_df)} images")
    return images, labels, names


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET WITH STRONG AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
class GlaucomaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.35, contrast=0.35,
                                   saturation=0.25, hue=0.08),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08),
                                    scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ARCHITECTURE — Deep CNN with Channel Attention (SE) + Spatial Attention
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class SpatialAttention(nn.Module):
    """Spatial attention — highlights where to look in feature maps."""
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with SE + Spatial attention."""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1  = ConvBNReLU(channels, channels)
        self.conv2  = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.se     = SEBlock(channels)
        self.spatial = SpatialAttention()
        self.drop   = nn.Dropout2d(dropout)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.se(out)
        out = self.spatial(out)
        return self.relu(out + x)


class GlaucomaNet(nn.Module):
    """
    Custom Deep CNN with:
      - 4 convolutional stages (increasing depth)
      - Residual blocks with SE + Spatial attention
      - Global Average Pooling
      - Deep classifier head with label smoothing
    """
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        # Stage 1 — 3 → 32
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 32, k=5, p=2),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        # Stage 2 — 32 → 64
        self.stage2 = nn.Sequential(
            ConvBNReLU(32, 64),
            ResidualBlock(64, dropout=0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
        # Stage 3 — 64 → 128
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128),
            ResidualBlock(128, dropout=0.1),
            ResidualBlock(128, dropout=0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
        )
        # Stage 4 — 128 → 256  (grad-cam hooks here)
        self.stage4 = nn.Sequential(
            ConvBNReLU(128, 256),
            ResidualBlock(256, dropout=0.15),
            ResidualBlock(256, dropout=0.15),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),  nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, num_classes)
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_cam_features(self, x):
        """Return stage4 feature maps + logits for Grad-CAM."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        feats = x                     # (B, 256, H', W')
        x = self.pool(x)
        logits = self.classifier(x)
        return feats, logits


# Label smoothing loss (helps regularize on tiny datasets)
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, targets):
        n_class = logits.size(1)
        smooth  = self.smoothing / (n_class - 1)
        one_hot = torch.full_like(logits, smooth)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_prob = F.log_softmax(logits, dim=1)
        if self.weight is not None:
            w = self.weight[targets]
            loss = -(one_hot * log_prob).sum(dim=1) * w
        else:
            loss = -(one_hot * log_prob).sum(dim=1)
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAIN / EVAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = 0
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return (total_loss / len(loader),
            correct / len(loader.dataset),
            np.array(all_preds),
            np.array(all_probs),
            np.array(all_labels))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
def compute_gradcam(model, img_tensor, target_class):
    """Compute Grad-CAM heatmap for one image."""
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_(False)

    feats, logits = model.get_cam_features(img_tensor)
    feats.retain_grad()
    feats_var = feats.requires_grad_(True)

    # recompute logits through pooling + classifier
    pooled  = model.pool(feats_var)
    logits2 = model.classifier(pooled)

    model.zero_grad()
    logits2[0, target_class].backward()

    grads   = feats_var.grad[0]               # (256, H', W')
    weights = grads.mean(dim=(1, 2))          # (256,)
    cam     = (weights[:, None, None] * feats_var[0]).sum(0)  # (H', W')
    cam     = F.relu(cam).detach().cpu().numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def denormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CROSS-VALIDATION TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def run_kfold(images, labels):
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    all_val_preds, all_val_probs, all_val_labels = [], [], []
    best_model_state = None
    best_val_auc     = 0

    print(f"\n{'='*55}")
    print(f"  {N_FOLDS}-Fold Cross-Validation Training")
    print(f"  Model: GlaucomaNet (Custom Deep CNN + Attention)")
    print(f"{'='*55}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(images, labels_arr), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──")
        train_imgs  = [images[i] for i in tr_idx]
        train_lbls  = labels_arr[tr_idx].tolist()
        val_imgs    = [images[i] for i in va_idx]
        val_lbls    = labels_arr[va_idx].tolist()

        # Weighted sampler to handle class imbalance
        counts  = Counter(train_lbls)
        weights = [1.0 / counts[l] for l in train_lbls]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_ds  = GlaucomaDataset(train_imgs, train_lbls, get_transforms(True))
        val_ds    = GlaucomaDataset(val_imgs,   val_lbls,   get_transforms(False))
        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        va_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        # Class-weighted loss
        n_total = len(train_lbls)
        cw = torch.tensor([n_total / (2 * counts[0]),
                            n_total / (2 * counts[1])],
                          dtype=torch.float).to(DEVICE)
        criterion = LabelSmoothingCE(smoothing=0.1, weight=cw)

        model     = GlaucomaNet(num_classes=2, dropout=0.45).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

        # Cosine annealing with warm restarts
        total_steps = EPOCHS * len(tr_loader)
        warmup = int(0.1 * total_steps)
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_fold_auc = 0
        best_fold_state = None
        tr_losses, va_losses, tr_accs, va_accs = [], [], [], []

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion,
                                              optimizer, scheduler)
            va_loss, va_acc, preds, probs, truths = evaluate(model, va_loader,
                                                              criterion)
            tr_losses.append(tr_loss); va_losses.append(va_loss)
            tr_accs.append(tr_acc);   va_accs.append(va_acc)

            if len(np.unique(truths)) > 1:
                fpr, tpr, _ = roc_curve(truths, probs)
                fold_auc = auc(fpr, tpr)
                if fold_auc >= best_fold_auc:
                    best_fold_auc   = fold_auc
                    best_fold_state = {k: v.clone()
                                       for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:02d}/{EPOCHS} | "
                      f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                      f"va_loss={va_loss:.4f} va_acc={va_acc:.3f}")

        # Restore best checkpoint for this fold
        model.load_state_dict(best_fold_state)
        _, va_acc, preds, probs, truths = evaluate(model, va_loader, criterion)
        f1 = f1_score(truths, preds, average='weighted')
        fpr, tpr, _ = roc_curve(truths, probs)
        fold_auc = auc(fpr, tpr)

        print(f"  → Best fold AUC={fold_auc:.4f}  Acc={va_acc:.4f}  F1={f1:.4f}")
        fold_results.append({'fold': fold, 'acc': va_acc,
                             'auc': fold_auc, 'f1': f1})
        all_val_preds.extend(preds)
        all_val_probs.extend(probs)
        all_val_labels.extend(truths)

        if fold_auc > best_val_auc:
            best_val_auc    = fold_auc
            best_model_state = best_fold_state

    return (fold_results, np.array(all_val_preds),
            np.array(all_val_probs), np.array(all_val_labels),
            best_model_state)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = {'glaucoma': '#E24B4A', 'normal': '#378ADD',
           'train': '#378ADD', 'val': '#E24B4A'}

def plot_fold_summary(fold_results):
    df = pd.DataFrame(fold_results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title, color in zip(
            axes, ['acc', 'f1', 'auc'],
            ['Accuracy', 'F1 Score', 'ROC-AUC'],
            ['#378ADD', '#3B6D11', '#E24B4A']):
        bars = ax.bar(df['fold'], df[col], color=color, alpha=0.85,
                      edgecolor='white', width=0.6)
        ax.axhline(df[col].mean(), color='k', linestyle='--', linewidth=1,
                   label=f'Mean={df[col].mean():.3f}')
        ax.set_ylim(0, 1.15); ax.set_xlabel('Fold'); ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
        for bar, v in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.3f}', ha='center', fontsize=9)
    plt.suptitle('5-Fold Cross-Validation Results — GlaucomaNet', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dl_fold_summary.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved: dl_fold_summary.png")


def plot_roc_and_cm(y_true, y_probs, y_preds):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='#E24B4A', lw=2.5,
             label=f'AUC = {roc_auc:.4f}')
    ax1.fill_between(fpr, tpr, alpha=0.08, color='#E24B4A')
    ax1.plot([0,1],[0,1],'--', color='gray', lw=1)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve (Cross-Val OOF)', fontsize=12)
    ax1.legend(fontsize=11); ax1.grid(alpha=0.3)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Normal','Glaucoma'],
                yticklabels=['Normal','Glaucoma'],
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax2.set_title('Confusion Matrix (OOF Predictions)', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dl_roc_cm.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved: dl_roc_cm.png")
    return roc_auc


def plot_probability_distribution(y_true, y_probs):
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls, lbl, color in [(0,'Normal','#378ADD'), (1,'Glaucoma','#E24B4A')]:
        probs = y_probs[y_true == cls]
        ax.hist(probs, bins=12, alpha=0.65, color=color,
                label=f'{lbl} (n={len(probs)})', edgecolor='white')
    ax.axvline(0.5, color='k', linestyle='--', linewidth=1.2, label='Threshold 0.5')
    ax.set_xlabel('Predicted Probability (Glaucoma)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Prediction Confidence Distribution', fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dl_prob_dist.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved: dl_prob_dist.png")


def plot_gradcam(model, val_images, val_labels, val_names, n_show=8):
    """Show Grad-CAM overlays for correct & incorrect predictions."""
    val_tf = get_transforms(False)
    model.eval()
    results = []
    for img, lab, name in zip(val_images, val_labels, val_names):
        tensor = val_tf(img)
        with torch.no_grad():
            feats, logits = model.get_cam_features(
                tensor.unsqueeze(0).to(DEVICE))
            prob  = F.softmax(logits, 1)[0, 1].item()
            pred  = int(prob >= 0.5)
        cam = compute_gradcam(model, tensor, pred)
        results.append({'img': img, 'label': lab, 'pred': pred,
                        'prob': prob, 'cam': cam, 'name': name})

    results.sort(key=lambda x: x['prob'], reverse=True)

    n_show = min(n_show, len(results))
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.5, 6))
    if n_show == 1: axes = axes[:, np.newaxis]

    class_names = ['Normal', 'Glaucoma']
    for col, r in enumerate(results[:n_show]):
        orig_np = np.array(r['img'].resize((IMG_SIZE, IMG_SIZE))) / 255.

        # Row 0: original
        axes[0, col].imshow(orig_np)
        color = 'green' if r['label'] == r['pred'] else 'red'
        axes[0, col].set_title(
            f"True: {class_names[r['label']]}\n"
            f"Pred: {class_names[r['pred']]} ({r['prob']:.2f})",
            fontsize=7.5, color=color)
        axes[0, col].axis('off')

        # Row 1: Grad-CAM overlay
        from PIL.Image import fromarray
        import matplotlib.cm as cm
        cam_resized = np.array(
            Image.fromarray((r['cam'] * 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.
        heatmap = cm.jet(cam_resized)[..., :3]
        overlay = 0.55 * orig_np + 0.45 * heatmap
        axes[1, col].imshow(overlay.clip(0, 1))
        axes[1, col].set_title('Grad-CAM', fontsize=7.5)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=9)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=9)
    plt.suptitle('Grad-CAM Visualizations — Model Attention on Optic Disc',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dl_gradcam.png", dpi=150, bbox_inches='tight')
    plt.close(); print("Saved: dl_gradcam.png")


def plot_architecture_summary(model):
    """Print parameter count and layer structure."""
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = [
        ('Stage 1', 'Conv5x5 → Conv3x3 → MaxPool',               '3 → 32'),
        ('Stage 2', 'Conv3x3 → ResBlock(SE+Spatial) → MaxPool',   '32 → 64'),
        ('Stage 3', 'Conv3x3 → 2×ResBlock → MaxPool',             '64 → 128'),
        ('Stage 4', 'Conv3x3 → 2×ResBlock',                       '128 → 256'),
        ('Pool',    'Global Average Pool',                          '256×H×W → 256'),
        ('Head',    'FC(256) → BN → FC(64) → BN → FC(2)',          '256 → 2'),
    ]
    print(f"\n{'─'*55}")
    print(f"  GlaucomaNet Architecture Summary")
    print(f"{'─'*55}")
    print(f"  {'Layer':<12} {'Operations':<40} {'Channels'}")
    print(f"  {'─'*12} {'─'*40} {'─'*10}")
    for name, ops, ch in layers:
        print(f"  {name:<12} {ops:<40} {ch}")
    print(f"{'─'*55}")
    print(f"  Total params : {total:,}")
    print(f"  Trainable    : {train:,}")
    print(f"{'─'*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  GLAUCOMA DETECTION — DEEP LEARNING PIPELINE")
    print("  Model: Custom CNN + SE Attention + Grad-CAM")
    print("=" * 55)

    # --- Load data ---
    label_df = load_labels(FILE_PATH)
    print(f"\nLabels: {label_df['label'].value_counts().to_dict()}")

    images, labels, names = load_images(FILE_PATH, label_df)
    if len(images) == 0:
        print("ERROR: No images loaded."); return

    # --- Show architecture ---
    demo_model = GlaucomaNet()
    plot_architecture_summary(demo_model)
    del demo_model

    # --- Cross-validation ---
    (fold_results, oof_preds, oof_probs,
     oof_labels, best_state) = run_kfold(images, labels)

    # --- Final metrics ---
    print(f"\n{'='*55}")
    print(f"  OUT-OF-FOLD (OOF) FINAL RESULTS")
    print(f"{'='*55}")
    print(classification_report(oof_labels, oof_preds,
                                 target_names=['Normal','Glaucoma']))

    acc = accuracy_score(oof_labels, oof_preds)
    f1  = f1_score(oof_labels, oof_preds, average='weighted')
    fpr, tpr, _ = roc_curve(oof_labels, oof_probs)
    oof_auc = auc(fpr, tpr)

    fold_df = pd.DataFrame(fold_results)
    print(f"\nPer-fold summary:")
    print(fold_df.to_string(index=False))
    print(f"\nMean Acc : {fold_df['acc'].mean():.4f} ± {fold_df['acc'].std():.4f}")
    print(f"Mean F1  : {fold_df['f1'].mean():.4f} ± {fold_df['f1'].std():.4f}")
    print(f"Mean AUC : {fold_df['auc'].mean():.4f} ± {fold_df['auc'].std():.4f}")
    print(f"OOF AUC  : {oof_auc:.4f}")

    # --- Plots ---
    plot_fold_summary(fold_results)
    plot_roc_and_cm(oof_labels, oof_probs, oof_preds)
    plot_probability_distribution(oof_labels, oof_probs)

    # Grad-CAM on all val images from last fold
    print("\nGenerating Grad-CAM visualizations...")
    final_model = GlaucomaNet().to(DEVICE)
    final_model.load_state_dict(best_state)
    plot_gradcam(final_model, images, labels, names, n_show=min(8, len(images)))

    # --- Save model ---
    torch.save({
        'model_state':   best_state,
        'oof_auc':       oof_auc,
        'mean_acc':      fold_df['acc'].mean(),
        'mean_f1':       fold_df['f1'].mean(),
        'img_size':      IMG_SIZE,
        'architecture':  'GlaucomaNet_SE_SpatialAttn',
    }, f"{OUT_DIR}/glaucoma_dl_model.pth")
    print(f"\nModel saved: glaucoma_dl_model.pth")
    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
