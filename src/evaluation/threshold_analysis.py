import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data.dataset import load_datasets


DATA_DIR = "data/processed/chest_xray"
MODEL_PATH = "models/disease_model.keras"


# Load test dataset
_, _, test_ds = load_datasets(DATA_DIR)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)


# Get predictions
y_true = []
y_prob = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy().flatten()
    )

    y_prob.extend(
        predictions.flatten()
    )


y_true = np.array(y_true)
y_prob = np.array(y_prob)


print("\n==============================")
print("THRESHOLD ANALYSIS")
print("==============================")

thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]


for threshold in thresholds:

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(f"\nThreshold: {threshold:.2f}")

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print("Confusion Matrix:")
    print(cm)