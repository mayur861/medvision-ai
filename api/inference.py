import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    if prediction >= 0.5:

        disease = "Pneumonia"
        confidence = prediction

    else:

        disease = "Normal"
        confidence = 1 - prediction

    return {
        "prediction": disease,
        "confidence": float(confidence)
    }