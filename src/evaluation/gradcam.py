import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "models/disease_model.keras"

IMAGE_PATH = (
    "data/processed/chest_xray/test/PNEUMONIA/"
    "0c1d54d9-0197-40d9-9b07-d99b1e8aa8db.png"
)

IMG_SIZE = (224, 224)

OUTPUT_DIR = "outputs/gradcam"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD IMAGE
# =========================================================

def load_xray(img_path):

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    img_array = image.img_to_array(img)

    # Keep original image for visualization
    original_image = img_array.copy()

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # -----------------------------------------------------
    # DenseNet preprocessing
    # -----------------------------------------------------

    img_array = tf.keras.applications.densenet.preprocess_input(
        img_array
    )

    return img_array, original_image


# =========================================================
# FIND GRAD-CAM LAYER
# =========================================================

def get_gradcam_layer(model):

    target_layer_name = "conv5_block16_concat"

    for layer in model.layers:

        if layer.name == target_layer_name:

            print(
                f"Grad-CAM layer found: {layer.name}"
            )

            return layer

    raise ValueError(
        f"Layer '{target_layer_name}' "
        "was not found in the model."
    )


# =========================================================
# GENERATE GRAD-CAM
# =========================================================

def generate_gradcam(
    model,
    img_array,
    gradcam_layer
):

    # -----------------------------------------------------
    # Create gradient model
    # -----------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            gradcam_layer.output,
            model.output
        ]
    )

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            {
                "image_input": img_array
            }
        )

        # Binary classification:
        # output = probability of Pneumonia

        pneumonia_probability = predictions[:, 0]

    # -----------------------------------------------------
    # Calculate gradients
    # -----------------------------------------------------

    gradients = tape.gradient(
        pneumonia_probability,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients are None. "
            "Check Grad-CAM layer connection."
        )

    # -----------------------------------------------------
    # Global Average Pooling
    # -----------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # -----------------------------------------------------
    # Weighted feature maps
    # -----------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # -----------------------------------------------------
    # ReLU
    # -----------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # -----------------------------------------------------
    # Normalize heatmap
    # -----------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    if max_value > 0:

        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()

    probability = float(
        pneumonia_probability.numpy()[0]
    )

    return heatmap, probability


# =========================================================
# SAVE GRAD-CAM IMAGE
# =========================================================

def save_gradcam(
    original_image,
    heatmap,
    prediction,
    predicted_class,
    output_path
):

    # -----------------------------------------------------
    # Resize heatmap
    # -----------------------------------------------------

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE
    ).numpy().squeeze()

    # -----------------------------------------------------
    # Convert original image
    # -----------------------------------------------------

    original_image = np.clip(
        original_image,
        0,
        255
    ).astype(
        np.uint8
    )

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    plt.figure(
        figsize=(16, 5)
    )

    # =====================================================
    # ORIGINAL IMAGE
    # =====================================================

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        "Original X-Ray",
        fontsize=14
    )

    plt.axis(
        "off"
    )

    # =====================================================
    # HEATMAP
    # =====================================================

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        heatmap_resized,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM Heatmap",
        fontsize=14
    )

    plt.axis(
        "off"
    )

    # =====================================================
    # OVERLAY
    # =====================================================

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        original_image
    )

    plt.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"{predicted_class} | "
        f"Probability: {prediction * 100:.2f}%",
        fontsize=14
    )

    plt.axis(
        "off"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("========================================")
    print("          GRAD-CAM ANALYSIS")
    print("========================================")
    print()

    # =====================================================
    # CHECK MODEL
    # =====================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            f"Model not found:\n{MODEL_PATH}"
        )

        return

    # =====================================================
    # CHECK IMAGE
    # =====================================================

    if not os.path.exists(
        IMAGE_PATH
    ):

        print(
            f"Image not found:\n{IMAGE_PATH}"
        )

        return

    print(
        "Image found successfully."
    )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    print()
    print(
        "Loading model..."
    )

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Model loaded successfully."
    )

    # =====================================================
    # GET GRAD-CAM LAYER
    # =====================================================

    print()
    print(
        "Finding Grad-CAM layer..."
    )

    gradcam_layer = get_gradcam_layer(
        model
    )

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    print()
    print(
        "Loading X-Ray..."
    )

    img_array, original_image = load_xray(
        IMAGE_PATH
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    print()
    print(
        "Running prediction..."
    )

    prediction = model.predict(
        {
            "image_input": img_array
        },
        verbose=0
    )[0][0]

    # =====================================================
    # CLASS
    # =====================================================

    if prediction >= 0.5:

        predicted_class = "PNEUMONIA"

    else:

        predicted_class = "NORMAL"

    # =====================================================
    # DISPLAY PREDICTION
    # =====================================================

    print()
    print("========================================")
    print("             PREDICTION")
    print("========================================")

    print(
        f"Pneumonia Probability : "
        f"{prediction:.4f}"
    )

    print(
        f"Probability           : "
        f"{prediction * 100:.2f}%"
    )

    print(
        f"Predicted Class       : "
        f"{predicted_class}"
    )

    # =====================================================
    # GENERATE GRAD-CAM
    # =====================================================

    print()
    print(
        "Generating Grad-CAM..."
    )

    heatmap, gradcam_probability = generate_gradcam(
        model,
        img_array,
        gradcam_layer
    )

    # =====================================================
    # SAVE RESULT
    # =====================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        "gradcam_result.png"
    )

    save_gradcam(
        original_image,
        heatmap,
        prediction,
        predicted_class,
        output_path
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print()
    print("========================================")
    print("        GRAD-CAM COMPLETED")
    print("========================================")

    print()
    print(
        f"Prediction : {predicted_class}"
    )

    print(
        f"Probability: {prediction * 100:.2f}%"
    )

    print()
    print(
        "Grad-CAM Result:"
    )

    print(
        output_path
    )

    print()
    print(
        "========================================"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()