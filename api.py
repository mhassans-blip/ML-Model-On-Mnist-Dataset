from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import uvicorn

app = FastAPI(
    title="MNIST ANN API",
    description="FastAPI deployment of the ANN trained on the MNIST dataset.",
    version="1.0.0"
)

MODEL_PATH = "mnist_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    model_error = str(e)


def preprocess_image(image: Image.Image):
    """Convert an uploaded digit image into the format used by the MNIST model."""
    image = image.convert("L")
    image = image.resize((28, 28))

    array = np.array(image).astype("float32") / 255.0

    # MNIST has a dark background and a bright digit.
    # If the uploaded image has a light background, invert it.
    if array.mean() > 0.5:
        array = 1.0 - array

    array = array.reshape(1, 28, 28)
    return array


@app.get("/")
def home():
    return {
        "message": "MNIST ANN API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "model_loaded": False, "error": model_error}
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model could not be loaded: {model_error}"
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:
        image = Image.open(file.file)
        processed = preprocess_image(image)

        probabilities = model.predict(processed, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        probabilities_dict = {
            str(i): float(probabilities[i]) for i in range(10)
        }

        return {
            "prediction": predicted_digit,
            "confidence": confidence,
            "probabilities": probabilities_dict
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")
