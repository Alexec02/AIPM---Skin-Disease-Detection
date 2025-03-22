import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to your dataset
dataset_path = r"D:\MS-AI\2ndSem\AIPM\Lab\AIPM---Skin-Disease-Detection\dataset"

# Define image size
image_size = (64, 64)

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=30,  # Random rotation between -30 and 30 degrees
    horizontal_flip=True,  # Random horizontal flip
)

# This will store image data and labels
metadata = []

def preprocess_images_from_folder(folder_path, class_name):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, image_size)
                img_resized_norm = img_resized / 255.0
                metadata.append({'image_data': img_resized_norm, 'class': class_name})
                img_resized_norm = np.expand_dims(img_resized_norm, axis=0)
                augmented_gen = datagen.flow(img_resized_norm, batch_size=1)
                for _ in range(1):
                    augmented_img = next(augmented_gen)[0]
                    metadata.append({'image_data': augmented_img, 'class': class_name})

def process_dataset(dataset_path):
    train_dir = os.path.join(dataset_path, 'train')
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        preprocess_images_from_folder(class_path, class_name)
    test_dir = os.path.join(dataset_path, 'test')
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        preprocess_images_from_folder(class_path, class_name)

def save_metadata_to_npz(metadata, filename='image_data.npz'):
    images = np.array([entry['image_data'] for entry in metadata])
    labels = np.array([entry['class'] for entry in metadata])
    np.savez_compressed(filename, images=images, labels=labels)
    print(f"Metadata saved to {filename}")

if __name__ == "__main__":
    process_dataset(dataset_path)
    save_metadata_to_npz(metadata)