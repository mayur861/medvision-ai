import os
import uuid
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model, Model


# =========================================================
# TensorFlow settings
# =========================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "disease_model.keras"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "gradcam"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="MedVision AI",
    description="Chest X-Ray Pneumonia Detection Research Prototype",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://medvision-ai-dusky.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Static folders
# =========================================================

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)

app.mount(
    "/gradcam",
    StaticFiles(directory=GRADCAM_DIR),
    name="gradcam"
)


# =========================================================
# Load model once
# =========================================================

print("Loading model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# =========================================================
# Grad-CAM layer
# =========================================================

GRADCAM_LAYER_NAME = "conv5_block16_concat"

try:
    gradcam_layer = model.get_layer(
        GRADCAM_LAYER_NAME
    )

    grad_model = Model(
        inputs=model.input,
        outputs=[
            gradcam_layer.output,
            model.output
        ]
    )

    GRADCAM_AVAILABLE = True

    print(
        f"Grad-CAM layer found: {GRADCAM_LAYER_NAME}"
    )

except Exception as e:

    grad_model = None
    GRADCAM_AVAILABLE = False

    print(
        "Grad-CAM unavailable:",
        str(e)
    )


# =========================================================
# Image preprocessing
# =========================================================

def preprocess_image(image: Image.Image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# Basic X-Ray validation
# =========================================================

def is_xray_like(image: Image.Image):

    image = image.convert("RGB")

    small = image.resize(
        (64, 64)
    )

    arr = np.array(
        small,
        dtype=np.float32
    )

    channel_difference = np.mean(
        np.abs(
            arr[:, :, 0] -
            arr[:, :, 1]
        )
    )

    channel_difference += np.mean(
        np.abs(
            arr[:, :, 1] -
            arr[:, :, 2]
        )
    )

    # X-rays are generally grayscale.
    return channel_difference < 15


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():

    return {
        "message": "MedVision AI API is running",
        "model": "DenseNet121",
        "task": "Chest X-Ray Pneumonia Detection",
        "status": "online"
    }


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "gradcam_available": GRADCAM_AVAILABLE
    }


# =========================================================
# Prediction
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    request_id = str(
        uuid.uuid4()
    )

    try:

        contents = await file.read()

        if not contents:
            return {
                "success": False,
                "error": "Empty file."
            }

        image = Image.open(
            __import__("io").BytesIO(contents)
        )

        image.load()

        # -------------------------------------------------
        # X-Ray validation
        # -------------------------------------------------

        if not is_xray_like(image):

            return {
                "success": False,
                "error": (
                    "Please upload a chest X-ray "
                    "image."
                ),
                "request_id": request_id
            }

        # -------------------------------------------------
        # Save uploaded image
        # -------------------------------------------------

        extension = ".png"

        original_name = (
            file.filename
            or "xray.png"
        )

        if "." in original_name:

            extension = (
                "." +
                original_name.rsplit(
                    ".",
                    1
                )[1].lower()
            )

        upload_filename = (
            f"{request_id}"
            f"_{original_name}"
        )

        upload_path = os.path.join(
            UPLOAD_DIR,
            upload_filename
        )

        image.save(
            upload_path
        )

        # -------------------------------------------------
        # Preprocess
        # -------------------------------------------------

        batch = preprocess_image(
            image
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        probability = model.predict(
            batch,
            verbose=0
        )[0][0]

        probability_percent = (
            float(probability) * 100
        )

        if probability >= 0.5:

            prediction = "PNEUMONIA"

        else:

            prediction = "NORMAL"

        confidence = (
            probability_percent
            if prediction == "PNEUMONIA"
            else 100 - probability_percent
        )

        # -------------------------------------------------
        # Cleanup
        # -------------------------------------------------

        del batch
        gc.collect()

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return {
            "success": True,
            "filename": original_name,
            "prediction": prediction,
            "pneumonia_probability": round(
                probability_percent,
                2
            ),
            "confidence": round(
                confidence,
                2
            ),
            "model": "DenseNet121",
            "task": (
                "Chest X-Ray "
                "Pneumonia Detection"
            ),
            "message": (
                "Research/decision-support "
                "prediction only. "
                "Not a clinical diagnosis."
            ),
            "gradcam_available": (
                GRADCAM_AVAILABLE
            ),
            "request_id": request_id,
            "uploaded_image": (
                f"/uploads/"
                f"{upload_filename}"
            )
        }

    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return {
            "success": False,
            "error": "Prediction failed.",
            "details": str(e),
            "request_id": request_id
        }


# =========================================================
# Grad-CAM
# =========================================================

@app.post("/gradcam")
async def generate_gradcam(
    file: UploadFile = File(...)
):

    request_id = str(
        uuid.uuid4()
    )

    if not GRADCAM_AVAILABLE:

        return {
            "success": False,
            "error": "Grad-CAM is unavailable."
        }

    try:

        contents = await file.read()

        image = Image.open(
            __import__("io").BytesIO(contents)
        )

        image.load()

        # -------------------------------------------------
        # Preprocess
        # -------------------------------------------------

        batch = preprocess_image(
            image
        )

        # -------------------------------------------------
        # Gradient calculation
        # -------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, predictions = (
                grad_model(batch)
            )

            pneumonia_score = predictions[:, 0]

        grads = tape.gradient(
            pneumonia_score,
            conv_outputs
        )

        # -------------------------------------------------
        # Global average pooling of gradients
        # -------------------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )

        conv_outputs = conv_outputs[0]

        pooled_grads = pooled_grads[0]

        heatmap = tf.reduce_sum(
            conv_outputs *
            pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        heatmap = (
            heatmap /
            (max_value + 1e-8)
        )

        heatmap = heatmap.numpy()

        # -------------------------------------------------
        # Convert heatmap to image
        # -------------------------------------------------

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap_image = Image.fromarray(
            heatmap
        ).resize(
            image.size
        )

        # -------------------------------------------------
        # Simple red/yellow heatmap
        # -------------------------------------------------

        heatmap_array = np.array(
            heatmap_image,
            dtype=np.float32
        ) / 255.0

        original_array = np.array(
            image.convert("RGB"),
            dtype=np.float32
        )

        # Create RGB heatmap
        heatmap_rgb = np.zeros_like(
            original_array
        )

        heatmap_rgb[:, :, 0] = (
            heatmap_array * 255
        )

        heatmap_rgb[:, :, 1] = (
            heatmap_array * 255
        )

        heatmap_rgb[:, :, 2] = 0

        # -------------------------------------------------
        # Overlay
        # -------------------------------------------------

        overlay = (
            0.65 * original_array +
            0.35 * heatmap_rgb
        )

        overlay = np.clip(
            overlay,
            0,
            255
        ).astype(
            np.uint8
        )

        overlay_image = Image.fromarray(
            overlay
        )

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        filename = (
            f"{request_id}_gradcam.png"
        )

        output_path = os.path.join(
            GRADCAM_DIR,
            filename
        )

        overlay_image.save(
            output_path
        )

        # -------------------------------------------------
        # Cleanup
        # -------------------------------------------------

        del batch
        del conv_outputs
        del grads
        del pooled_grads
        del heatmap
        del heatmap_array
        del original_array
        del heatmap_rgb
        del overlay

        gc.collect()

        return {
            "success": True,
            "gradcam_image": (
                f"/gradcam/{filename}"
            ),
            "request_id": request_id
        }

    except Exception as e:

        print(
            "Grad-CAM error:",
            repr(e)
        )

        gc.collect()

        return {
            "success": False,
            "error": "Grad-CAM generation failed.",
            "details": str(e),
            "request_id": request_id
        }