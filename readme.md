# Skin Disease Detection using CNN

### Authors
- Ayesha Munir
- Alejandro Esperón Couceiro

## Overview
This project aims to detect skin diseases using a Convolutional Neural Network (CNN) model. The dataset consists of images of various skin conditions, and the model is trained to classify these images into different categories.

## Project Structure
- `data_preprocessing.py`: Script for preprocessing the dataset and saving the processed data.
- `cnn_model.py`: Defines the CNN model architecture.
- `cnn_model_grid_search.py`: Performs grid search to find the best hyperparameters for the CNN model.
- `main.ipynb`: Lload data, perform grid search, train the model, and save the trained models.

## Instructions

### Step 1: Data Preprocessing
Before running the main notebook, you must preprocess the dataset. This step involves resizing, normalizing, and augmenting the images, and saving the processed data to a file.

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command:
    ```bash
    python data_preprocessing.py
    ```

This will preprocess the images and save the metadata to `image_data.npz`.

### Step 2: Training the Model
After preprocessing the data, you can run the main notebook to train the model and perform hyperparameter tuning.

1. Open the `main.ipynb` file in a Jupyter notebook or any Python IDE.
2. Execute the cells to load the data, perform grid search, train the model, and save the trained models.
