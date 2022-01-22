import tensorflow as tf
from src import config


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


class Model:

    def __init__(self, tp_obj):
        self.tp_obj = tp_obj
        self.ctc_obj = CTCLayer(name="ctc_loss")
        pass

    def handwriting_recognition(self):
        image = tf.keras.Input(name="image", shape=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_CHANNELS))
        label = tf.keras.Input(name="label", shape=(None,))

        x = tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   kernel_initializer='he_normal',
                                   padding="same",
                                   name='Conv2D_1')(image)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPooling2D_1")(x)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   kernel_initializer='he_normal',
                                   padding="same",
                                   name='Conv2D_2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPooling2D_2")(x)

        new_shape = ((config.IMAGE_WIDTH // 4), (config.IMAGE_HEIGHT // 4) * 64)
        x = tf.keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        x = tf.keras.layers.Dense(
            len(self.tp_obj.char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2")(x)

        # Add CTC layer for calculating CTC loss at each step

        output = CTCLayer(name="ctc_loss")(label, x)

        # Define the model
        model = tf.keras.models.Model(
            inputs=[image, label], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = tf.keras.optimizers.Adam()
        # Compile the model and return
        model.compile(optimizer=opt)
        return model
