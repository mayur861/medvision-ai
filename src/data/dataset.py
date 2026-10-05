import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def load_datasets(data_dir):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="binary",
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="binary",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="binary",
        shuffle=False
    )

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds