import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.data.dataset import load_datasets
from tensorflow.keras.models import load_model


DATA_DIR = "data/processed/chest_xray"

_, _, test_ds = load_datasets(DATA_DIR)

model = load_model("models/disease_model.keras")

y_true = []
y_prob = []


for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_prob.extend(predictions.flatten())


y_true = np.array(y_true)
y_prob = np.array(y_prob)

y_pred = (y_prob >= 0.5).astype(int)


print(
    classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "PNEUMONIA"]
    )
)


print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print(
    "ROC-AUC:",
    roc_auc_score(y_true, y_prob)
)