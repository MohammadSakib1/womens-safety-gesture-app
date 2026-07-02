# from __future__ import annotations

# import tensorflow as tf


# class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
#     def __init__(self, gamma: float = 2.0, alpha: float = 0.25, name: str = "sparse_focal_loss"):
#         super().__init__(name=name)
#         self.gamma = gamma
#         self.alpha = alpha

#     def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
#         y_true = tf.cast(y_true, tf.int32)
#         y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
#         y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
#         ce = -y_true_one_hot * tf.math.log(y_pred)
#         weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
#         loss = weight * ce
#         return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))



from __future__ import annotations

import tensorflow as tf


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, name: str = "sparse_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        ce = -y_true_one_hot * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        loss = weight * ce

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))