import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# 1. Load Patient Excel Records
excel_path = os.path.join("data", "Data on OCT and Fundus Images", "Patient Record-final.xlsx")

if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    print("--- Excel Metadata Summary ---")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
else:
    print(f"Excel file not found at {excel_path}")

# 2. Inspect Image Files
sample_image_path = os.path.join(
    "data", 
    "Data on OCT and Fundus Images", 
    "P 1", 
    "109151_20150910_080317_B-scan_R_001.jpg"
)

if os.path.exists(sample_image_path):
    # Load image using OpenCV
    img = cv2.imread(sample_image_path)
    print(f"\n--- Sample Image Details ---")
    print(f"Path: {sample_image_path}")
    print(f"Dimensions (Height, Width, Channels): {img.shape}")
    
    # Display the image using Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Sample OCT B-Scan Image")
    plt.axis("off")
    plt.show()
else:
    print(f"Sample image not found at {sample_image_path}")