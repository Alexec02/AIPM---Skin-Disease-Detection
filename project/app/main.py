import pyheif
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
model = tf.keras.models.load_model('xception_model.h5')
disease_classes = [
    "Acne and Rosacea",
    "Actinic Keratosis, Basal Cell Carcinoma, and Malignant Lesions",
    "Atopic Dermatitis",
    "Bullous Diseases",
    "Cellulitis, Impetigo, and Bacterial Infections",
    "Eczema",
    "Exanthems and Drug Eruptions",
    "Hair Loss (Alopecia) and Other Hair Diseases",
    "Herpes, HPV, and Other STDs",
    "Light Diseases and Pigmentation Disorders",
    "Lupus and Connective Tissue Diseases",
    "Melanoma, Skin Cancer, Nevi, and Moles",
    "Nail Fungus and Other Nail Diseases",
    "Poison Ivy and Contact Dermatitis",
    "Psoriasis, Lichen Planus, and Related Diseases",
    "Scabies, Lyme Disease, and Infestations",
    "Seborrheic Keratoses and Benign Tumors",
    "Systemic Diseases",
    "Tinea, Ringworm, Candidiasis, and Fungal Infections",
    "Urticaria (Hives)",
    "Vascular Tumors",
    "Vasculitis",
    "Warts, Molluscum, and Viral Infections"
]

# Image transformation to match the model's input requirements
def prepare_image(img_bytes):
    # Check if the image is in HEIC format
    try:
        heif_file = pyheif.read(img_bytes)  # Try reading as HEIC
        img = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data
        )
    except:
        # If not HEIC, proceed with normal processing
        img = Image.open(io.BytesIO(img_bytes))

    # Check if the image has an alpha channel and convert it to RGB (removes the alpha channel)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to 128x128 as per the model's requirement
    img = img.resize((71, 71))  # Resize to model input size (128x128)
    
    # Convert image to numpy array and expand the dimensions
    img_array = np.array(img)
    
    # Add batch dimension (shape will be (1, 128, 128, 3))
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image according to the model's preprocessing
    img_array = preprocess_input(img_array)
    
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

        return {"message": "Image processed successfully", "predicted_class": disease_classes[predicted_class]}

    except Exception as e:
        return {"error": str(e)}
