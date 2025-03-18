# Import necessary libraries for preprocessing
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size
image_size = (64, 64)

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=30,  # Random rotation between -30 and 30 degrees
    horizontal_flip=True,  # Random horizontal flip
)

# This will store image paths, labels, and augmented pixel data
metadata = []

def preprocess_images_from_folder(folder_path, datagen, class_name):
    """
    Process all images from a folder with the given transformations (resize, augmentations)
    and store the image paths and labels into the metadata list.
    :param folder_path: The path to the folder containing images
    :param datagen: ImageDataGenerator for transformations
    :param class_name: The class name (folder name) of the images
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)

                # Read the image using OpenCV
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB

                # Resize the image to the target size
                img_resized = cv2.resize(img, image_size)

                # Normalize the image
                img_resized_norm = img_resized / 255.0

                # Add to metadata
                metadata.append({
                    'label': class_name,
                    'image_data': img_resized_norm
                })

                # Apply augmentation
                img_resized_norm = np.expand_dims(img_resized_norm, axis=0)  # Add batch dimension
                augmented_gen = datagen.flow(img_resized_norm, batch_size=1)

                for _ in range(1):  # Generate augmented images
                    augmented_img = next(augmented_gen)[0]

                    # Add augmented image to metadata
                    metadata.append({
                        'label': class_name,
                        'image_data': augmented_img
                    })

def process_dataset(dataset_path, datagen):
    """
    Loop through the 'train' and 'test' directories and preprocess all images
    :param dataset_path: The root path of the dataset
    :param datagen: ImageDataGenerator for transformations
    """
    # Process the 'train' folder
    train_dir = os.path.join(dataset_path, 'train')
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        preprocess_images_from_folder(class_path, datagen, class_name)

    # Process the 'test' folder
    test_dir = os.path.join(dataset_path, 'test')
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        preprocess_images_from_folder(class_path, datagen, class_name)

# Collect metadata and save to CSV
def save_metadata_to_csv(metadata):
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('image_metadata.csv', index=False)
    print("Metadata saved to 'image_metadata.csv'")

# Run the preprocessing
dataset_path = r"C:\Users\Alex\Desktop\AIPM\dataset"
process_dataset(dataset_path, datagen)

# Save metadata to CSV
save_metadata_to_csv(metadata)
