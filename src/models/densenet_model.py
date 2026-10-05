
import tensorflow as tf

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.models import Model


IMG_SIZE = (224, 224)


def build_model():

    # Input layer
    inputs = Input(
        shape=(224, 224, 3),
        name="image_input"
    )

    # ImageNet pretrained DenseNet121
    # base_model = DenseNet121(
    #     include_top=False,
    #     weights="imagenet",
    #     input_tensor=inputs
    # )
    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="image_input"
    )

    x = preprocess_input(inputs)

    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    base_model.trainable = False

    x = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="prediction"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Pneumonia_DenseNet121"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Feature extraction
    x = base_model.output

    # Convert feature maps into vector
    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # Fully connected layer
    x = Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)

    # Prevent overfitting
    x = Dropout(
        0.4,
        name="dropout"
    )(x)

    # Binary classification
    outputs = Dense(
        1,
        activation="sigmoid",
        name="prediction"
    )(x)

    # Create model
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Pneumonia_DenseNet121"
    )

    # Compile
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(
    #         learning_rate=0.0001
    #     ),

    #     loss="binary_crossentropy",

    #     metrics=[
    #         "accuracy",

    #         tf.keras.metrics.Precision(
    #             name="precision"
    #         ),

    #         tf.keras.metrics.Recall(
    #             name="recall"
    #         ),

    #         tf.keras.metrics.AUC(
    #             name="auc"
    #         )
    #     ]
    # )
    model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

    return model

