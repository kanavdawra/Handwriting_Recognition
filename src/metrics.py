import tensorflow as tf
import numpy as np


def calculate_edit_distance(sources, predictions, max_len):
    sources_s_tensor = tf.cast(tf.sparse.from_dense(sources), dtype=tf.int64)
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = tf.keras.backend.ctc_decode(predictions, input_length=input_len, greedy=True)[0][0] \
        [:, :max_len]
    predictions_s_tensor = tf.cast(tf.sparse.from_dense(predictions_decoded), dtype=tf.int64)
    edit_distances = tf.edit_distance(predictions_s_tensor, sources_s_tensor, normalize=False)
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(tf.keras.callbacks.Callback):
    def __init__(self, pred_model, valid_dataset, max_len):
        super().__init__()
        self.prediction_model = pred_model
        self.valid_dataset = valid_dataset
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        validation_images = []
        validation_labels = []

        for batch in self.valid_dataset:
            validation_images.append(batch["image"])
            validation_labels.append(batch["label"])
        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions, self.max_len).numpy())

        print(
            f"   Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )
