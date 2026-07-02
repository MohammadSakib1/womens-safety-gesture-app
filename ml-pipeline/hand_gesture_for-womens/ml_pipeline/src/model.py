# from __future__ import annotations

# import tensorflow as tf


# class AttentionGate(tf.keras.layers.Layer):
#     def __init__(self, units: int, **kwargs):
#         super().__init__(**kwargs)
#         self.dense_1 = tf.keras.layers.Dense(units, activation="relu")
#         self.dense_2 = tf.keras.layers.Dense(units, activation="sigmoid")

#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         weights = self.dense_1(inputs)
#         weights = self.dense_2(weights)
#         return inputs * weights


# def residual_dense_block(
#     x: tf.Tensor,
#     units: int,
#     dropout_rate: float,
#     l2_weight: float,
# ) -> tf.Tensor:
#     shortcut = x

#     if x.shape[-1] != units:
#         shortcut = tf.keras.layers.Dense(
#             units,
#             kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
#         )(shortcut)

#     out = tf.keras.layers.Dense(
#         units,
#         activation="relu",
#         kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
#     )(x)
#     out = tf.keras.layers.BatchNormalization()(out)
#     out = tf.keras.layers.Dropout(dropout_rate)(out)

#     out = tf.keras.layers.Dense(
#         units,
#         activation="relu",
#         kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
#     )(out)
#     out = tf.keras.layers.BatchNormalization()(out)

#     out = tf.keras.layers.Add()([shortcut, out])
#     out = tf.keras.layers.Activation("relu")(out)
#     out = tf.keras.layers.Dropout(dropout_rate)(out)
#     return out


# def build_resmlp_feature_fusion_model(
#     input_dim: int,
#     raw_dim: int = 63,
#     num_classes: int = 5,
#     dropout_rate: float = 0.30,
#     l2_weight: float = 1e-4,
# ) -> tf.keras.Model:
#     if input_dim <= raw_dim:
#         raise ValueError("input_dim must be greater than raw_dim.")

#     inputs = tf.keras.layers.Input(shape=(input_dim,), name="gesture_features")

#     raw_branch = tf.keras.layers.Lambda(
#         lambda t: t[:, :raw_dim],
#         name="raw_branch_slice",
#     )(inputs)

#     geo_branch = tf.keras.layers.Lambda(
#         lambda t: t[:, raw_dim:],
#         name="geo_branch_slice",
#     )(inputs)

#     raw_branch = tf.keras.layers.Dense(
#         256,
#         activation="relu",
#         kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
#     )(raw_branch)
#     raw_branch = tf.keras.layers.BatchNormalization()(raw_branch)
#     raw_branch = tf.keras.layers.Dropout(dropout_rate)(raw_branch)

#     geo_branch = tf.keras.layers.Dense(
#         128,
#         activation="relu",
#         kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
#     )(geo_branch)
#     geo_branch = tf.keras.layers.BatchNormalization()(geo_branch)
#     geo_branch = tf.keras.layers.Dropout(dropout_rate)(geo_branch)

#     x = tf.keras.layers.Concatenate(name="feature_fusion")([raw_branch, geo_branch])
#     x = residual_dense_block(x, 256, dropout_rate, l2_weight)
#     x = residual_dense_block(x, 128, dropout_rate, l2_weight)
#     x = AttentionGate(units=128, name="attention_gate")(x)

#     x = tf.keras.layers.Dense(
#         128,
#         activation="relu",
#         kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
#     )(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(dropout_rate)(x)

#     outputs = tf.keras.layers.Dense(
#         num_classes,
#         activation="softmax",
#         name="gesture_output",
#     )(x)

#     return tf.keras.Model(inputs=inputs, outputs=outputs, name="resmlp_feature_fusion")

from __future__ import annotations

import tensorflow as tf


class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_1 = tf.keras.layers.Dense(units, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        weights = self.dense_1(inputs)
        weights = self.dense_2(weights)
        return inputs * weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, start: int, end: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.end is None:
            return inputs[:, self.start:]
        return inputs[:, self.start:self.end]

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config


def residual_dense_block(
    x: tf.Tensor,
    units: int,
    dropout_rate: float,
    l2_weight: float,
) -> tf.Tensor:
    shortcut = x

    if x.shape[-1] != units:
        shortcut = tf.keras.layers.Dense(
            units,
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
        )(shortcut)

    out = tf.keras.layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(x)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = tf.keras.layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(out)
    out = tf.keras.layers.BatchNormalization()(out)

    out = tf.keras.layers.Add()([shortcut, out])
    out = tf.keras.layers.Activation("relu")(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    return out


def build_resmlp_feature_fusion_model(
    input_dim: int,
    raw_dim: int = 63,
    num_classes: int = 5,
    dropout_rate: float = 0.35,
    l2_weight: float = 1e-4,
) -> tf.keras.Model:
    if input_dim <= raw_dim:
        raise ValueError("input_dim must be greater than raw_dim.")

    inputs = tf.keras.layers.Input(shape=(input_dim,), name="gesture_features")

    raw_branch = SliceLayer(0, raw_dim, name="raw_branch_slice")(inputs)
    geo_branch = SliceLayer(raw_dim, None, name="geo_branch_slice")(inputs)

    raw_branch = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(raw_branch)
    raw_branch = tf.keras.layers.BatchNormalization()(raw_branch)
    raw_branch = tf.keras.layers.Dropout(dropout_rate)(raw_branch)

    geo_branch = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(geo_branch)
    geo_branch = tf.keras.layers.BatchNormalization()(geo_branch)
    geo_branch = tf.keras.layers.Dropout(dropout_rate * 0.8)(geo_branch)

    x = tf.keras.layers.Concatenate(name="feature_fusion")([raw_branch, geo_branch])

    x = residual_dense_block(x, 256, dropout_rate, l2_weight)
    x = residual_dense_block(x, 256, dropout_rate, l2_weight)
    x = residual_dense_block(x, 128, dropout_rate, l2_weight)

    x = AttentionGate(units=128, name="attention_gate")(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="gesture_output",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="resmlp_feature_fusion")