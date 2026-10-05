import tensorflow as tf

from src.data.dataset import load_datasets
from src.models.densenet_model import build_model


DATA_DIR = "data/processed/chest_xray"


train_ds, val_ds, test_ds = load_datasets(DATA_DIR)

model = build_model()

model.summary()


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        "models/disease_model.keras",
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2
    )
]


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)