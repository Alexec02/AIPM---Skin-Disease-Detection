from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

app = FastAPI()

# Load the model - need to change this
model = tf.keras.models.load_model('model.h5')

# Image transformation to match the model's input requirements
def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize based on model's requirements
    return img_array

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await image.read()

        # Prepare the image for prediction
        img_array = prepare_image(image_bytes)

        # Get model prediction
        predictions = model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(predictions, axis=1).item()

        return {"message": "Image processed successfully", "predicted_class": predicted_class}

    except Exception as e:
        return {"error": str(e)}
