from __future__ import annotations


def build_mobilenetv2(num_classes=5, input_size=224, dropout=0.3):
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ], name="geometric_augmentation")
    inputs = tf.keras.Input((input_size, input_size, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="flower_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_flowers"), base


def unfreeze_last_layers(base_model, count=30):
    import tensorflow as tf
    base_model.trainable = True
    for layer in base_model.layers[:-count]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
