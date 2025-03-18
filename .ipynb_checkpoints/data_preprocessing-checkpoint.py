import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Function to augment images (rotate and flip)
def augment_image(img):
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip

    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # Vertical flip

    return img

# Function to load data from CSV and process images
def load_data(csv_path, img_dir, image_size=(128, 128)):
    df = pd.read_csv(csv_path)
    
    # Ensure the columns are correct
    print("CSV Columns:", df.columns)

    X = []
    y = []

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['Image Path'])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        img = cv2.resize(img, image_size)

        # Apply augmentations (you can adjust the count or parameters)
        for _ in range(3):  # Augment the image 3 times
            augmented_img = augment_image(img.copy())
            X.append(augmented_img)
            y.append([row['KA'], row['KB'], row['CA'], row['DT'], row['M'], row['L'], row['MN']])

    # Normalize the images
    X = np.array(X).astype('float32') / 255.0  
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)
