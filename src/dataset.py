import os, re, zipfile
from io import BytesIO
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GlaucomaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_transforms(img_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def load_data_from_zip(zip_path):
    if os.path.isdir(zip_path):
        xlsx_files = []
        for root, _, files in os.walk(zip_path):
            xlsx_files.extend(os.path.join(root, name)
                              for name in files if name.lower().endswith('.xlsx'))
        if not xlsx_files:
            raise FileNotFoundError(f"No .xlsx label file found in: {zip_path}")
        df = pd.read_excel(xlsx_files[0], header=None)
    else:
        with zipfile.ZipFile(zip_path) as z:
            xlsx = [n for n in z.namelist() if n.lower().endswith('.xlsx')][0]
            with z.open(xlsx) as f:
                df = pd.read_excel(f, header=None)

    data = df.iloc[1:].copy()
    data.columns = range(data.shape[1])
    records = []
    for _, row in data.iterrows():
        name = str(row[0]).strip().strip("'").strip()
        if not name or name == 'nan': continue
        cdrs, labels = [], []
        for c in [1, 3, 5, 7]:
            try: cdrs.append(float(row[c]))
            except: pass
            lab = str(row[c+1]).strip().lower() if pd.notna(row[c+1]) else ''
            if lab in ('yes', 'no', 'suspect'): labels.append(lab)
        if not labels: continue
        vote = Counter(labels).most_common(1)[0][0]
        binary = 1 if vote in ('yes', 'suspect') else 0
        if not name.lower().endswith('.jpg'): name += '.jpg'
        records.append({'image_name': name, 'label': binary})

    label_df = pd.DataFrame(records)
    images, labels, names = [], [], []
    if os.path.isdir(zip_path):
        image_map = {}
        for root, _, files in os.walk(zip_path):
            for name in files:
                if name.lower().endswith('.jpg'):
                    image_map[name] = os.path.join(root, name)

        def open_image(path):
            return Image.open(path).convert('RGB')
    else:
        archive = zipfile.ZipFile(zip_path)
        image_map = {os.path.basename(name): name for name in archive.namelist()
                     if name.lower().endswith('.jpg')}

        def open_image(path):
            with archive.open(path) as file:
                return Image.open(BytesIO(file.read())).convert('RGB')

    try:
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
    finally:
        if not os.path.isdir(zip_path):
            archive.close()
    return images, labels, names