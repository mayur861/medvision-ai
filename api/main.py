import os
import uuid
import gc
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model


# =========================================================
# RUNTIME / MEMORY CONFIG
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Reduce CPU memory pressure on small Render instances
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "models/disease_model.keras"

IMG_SIZE = (224, 224)

UPLOAD_DIR = "outputs/uploads"
GRADCAM_DIR = "outputs/gradcam"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Human Disease Diagnosis API",
    description="Chest X-Ray Pneumonia Detection API",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STATIC FILES
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
# LOAD MODEL ONCE
# =========================================================

print("Loading DenseNet121 model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# =========================================================
# GRAD-CAM LAYER
# =========================================================

GRADCAM_LAYER_NAME = "conv5_block16_concat"

try:
    gradcam_layer = model.get_layer(
        GRADCAM_LAYER_NAME
    )

    print(
        f"Grad-CAM layer found: {GRADCAM_LAYER_NAME}"
    )

except Exception:
    gradcam_layer = None

    print(
        "Warning: Grad-CAM layer not found."
    )


# =========================================================
# CREATE GRAD-CAM MODEL ONLY ONCE
# =========================================================

grad_model = None

if gradcam_layer is not None:

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                gradcam_layer.output,
                model.output
            ]
        )

        print("Grad-CAM model created successfully.")

    except Exception as e:

        grad_model = None

        print(
            "Warning: Could not create Grad-CAM model:",
            e
        )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Human Disease Diagnosis API",
        "status": "running",
        "model": "DenseNet121",
        "task": "Pneumonia Detection",
        "gradcam": grad_model is not None
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_available": grad_model is not None
    }


# =========================================================
# BASIC CHEST X-RAY VALIDATION
# =========================================================

def is_xray_like(image):

    """
    Basic color-based input guard.

    This is NOT a medical-grade X-ray detector.
    """

    img = image.convert("RGB")
    img = img.resize((128, 128))

    arr = np.asarray(
        img,
        dtype=np.float32
    ) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    channel_difference = (
        np.abs(r - g)
        + np.abs(g - b)
        + np.abs(r - b)
    ) / 3.0

    color_score = float(
        np.mean(channel_difference)
    )

    if color_score > 0.08:
        return False

    return True


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_bytes):

    img = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    img = img.resize(
        IMG_SIZE
    )

    img_array = np.asarray(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # IMPORTANT:
    # Same preprocessing used during training.
    # No DenseNet preprocess_input here.
    return img_array


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(
    image_bytes,
    output_path
):

    if grad_model is None:
        return False

    img = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    original_size = img.size

    img = img.resize(
        IMG_SIZE
    )

    img_array = np.asarray(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # -----------------------------------------
    # Tensor
    # -----------------------------------------

    img_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32
    )

    # -----------------------------------------
    # Gradient calculation
    # -----------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_tensor,
            training=False
        )

        class_channel = predictions[:, 0]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    if grads is None:
        return False

    # -----------------------------------------
    # Global average pooling
    # -----------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # -----------------------------------------
    # Normalize
    # -----------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )

    heatmap = heatmap.numpy()

    # -----------------------------------------
    # Resize heatmap
    # -----------------------------------------

    heatmap_img = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_img = heatmap_img.resize(
        original_size
    )

    heatmap_array = np.asarray(
        heatmap_img,
        dtype=np.uint8
    )

    # -----------------------------------------
    # Heatmap RGB
    # -----------------------------------------

    heatmap_rgb = np.zeros(
        (
            heatmap_array.shape[0],
            heatmap_array.shape[1],
            3
        ),
        dtype=np.uint8
    )

    heatmap_rgb[:, :, 0] = heatmap_array

    heatmap_rgb[:, :, 1] = (
        heatmap_array * 0.5
    ).astype(np.uint8)

    heatmap_rgb[:, :, 2] = 0

    heatmap_rgb = Image.fromarray(
        heatmap_rgb
    )

    # -----------------------------------------
    # Overlay
    # -----------------------------------------

    original_img = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    overlay = Image.blend(
        original_img,
        heatmap_rgb,
        alpha=0.40
    )

    overlay.save(
        output_path,
        format="PNG"
    )

    # -----------------------------------------
    # Cleanup tensors
    # -----------------------------------------

    del img_tensor
    del img_array
    del conv_outputs
    del pooled_grads
    del grads
    del heatmap

    gc.collect()

    return True


# =========================================================
# PREDICTION API
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # =====================================================
    # FILE EXTENSION
    # =====================================================

    allowed_extensions = (
        ".png",
        ".jpg",
        ".jpeg"
    )

    filename = (
        file.filename or ""
    ).lower()

    if not filename.endswith(
        allowed_extensions
    ):

        return {
            "success": False,
            "error":
                "Only PNG, JPG and JPEG images are allowed."
        }

    # =====================================================
    # READ FILE
    # =====================================================

    image_bytes = await file.read()

    if not image_bytes:

        return {
            "success": False,
            "error":
                "Uploaded image is empty."
        }

    # =====================================================
    # OPEN IMAGE
    # =====================================================

    try:

        image = Image.open(
            BytesIO(image_bytes)
        )

        image.load()

    except Exception:

        return {
            "success": False,
            "error":
                "Invalid or corrupted image file."
        }

    # =====================================================
    # BASIC X-RAY VALIDATION
    # =====================================================

    if not is_xray_like(image):

        return {
            "success": False,
            "error":
                "This image does not appear to be a chest X-ray. "
                "Please upload a valid chest X-ray image.",
            "error_type":
                "INVALID_XRAY"
        }

    # =====================================================
    # PREPROCESS
    # =====================================================

    try:

        img_array = preprocess_image(
            image_bytes
        )

    except Exception:

        return {
            "success": False,
            "error":
                "Unable to process the uploaded image."
        }

    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        prediction_result = model.predict(
            img_array,
            batch_size=1,
            verbose=0
        )

        probability = float(
            prediction_result[0][0]
        )

        del prediction_result
        del img_array

        gc.collect()

    except Exception as e:

        return {
            "success": False,
            "error":
                f"Model prediction failed: {str(e)}"
        }

    # =====================================================
    # CLASS
    # =====================================================

    if probability >= 0.5:

        predicted_class = "PNEUMONIA"

    else:

        predicted_class = "NORMAL"

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = (
        probability
        if predicted_class == "PNEUMONIA"
        else 1.0 - probability
    )

    # =====================================================
    # UNIQUE ID
    # =====================================================

    request_id = str(
        uuid.uuid4()
    )

    original_filename = os.path.basename(
        file.filename or "uploaded_image.png"
    )

    safe_filename = (
        request_id
        + "_"
        + original_filename
    )

    # =====================================================
    # SAVE UPLOADED IMAGE
    # =====================================================

    save_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    try:

        with open(
            save_path,
            "wb"
        ) as f:

            f.write(
                image_bytes
            )

    except Exception as e:

        print(
            "Upload save error:",
            e
        )

    # =====================================================
    # GRAD-CAM
    # =====================================================

    gradcam_filename = (
        request_id
        + "_gradcam.png"
    )

    gradcam_path = os.path.join(
        GRADCAM_DIR,
        gradcam_filename
    )

    gradcam_generated = False

    try:

        gradcam_generated = generate_gradcam(
            image_bytes,
            gradcam_path
        )

    except Exception as e:

        print(
            "Grad-CAM error:",
            e
        )

        gradcam_generated = False

        # Make sure request-level garbage is released
        gc.collect()

    # =====================================================
    # RESPONSE
    # =====================================================

    response = {

        "success": True,

        "filename":
            file.filename,

        "prediction":
            predicted_class,

        "pneumonia_probability":
            round(
                probability * 100,
                2
            ),

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "model":
            "DenseNet121",

        "task":
            "Chest X-Ray Pneumonia Detection",

        "uploaded_image":
            f"/uploads/{safe_filename}",

        "gradcam_available":
            gradcam_generated,

        "message":
            "Research/decision-support prediction only. "
            "Not a clinical diagnosis."
    }

    if gradcam_generated:

        response["gradcam_image"] = (
            f"/gradcam/{gradcam_filename}"
        )

    # Final cleanup
    gc.collect()

    return response