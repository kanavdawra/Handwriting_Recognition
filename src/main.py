import pandas as pd
from src import config
import tensorflow as tf
from src import dataset
from src import metrics
from src.model import *


def train():
    train = pd.read_csv(config.BASE_DATA_DIR + r'/written_name_train_v2.csv')
    valid = pd.read_csv(config.BASE_DATA_DIR + r'/written_name_validation_v2.csv')
    test = pd.read_csv(config.BASE_DATA_DIR + r'/written_name_test_v2.csv')

    train.dropna(axis=0, inplace=True, how='any')
    train.reset_index(drop=True, inplace=True)
    valid.dropna(axis=0, inplace=True, how='any')
    valid.reset_index(drop=True, inplace=True)
    test.dropna(inplace=True)
    test.reset_index(drop=True, inplace=True)

    train.FILENAME = config.BASE_DATA_DIR + config.TRAIN_DATA_DIR + train.FILENAME
    valid.FILENAME = config.BASE_DATA_DIR + config.VALIDATION_DATA_DIR + valid.FILENAME
    test.FILENAME = config.BASE_DATA_DIR + config.TEST_DATA_DIR + test.FILENAME

    ds_obj = dataset.ClassificationDataset(train.IDENTITY)

    train_dataset = ds_obj.get_dataset(train.FILENAME, train.IDENTITY)
    valid_dataset = ds_obj.get_dataset(valid.FILENAME, valid.IDENTITY)

    tp_obj = ds_obj.tp_obj
    model_obj = Model(tp_obj)

    model = model_obj.handwriting_recognition()
    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    edit_distance_callback = metrics.EditDistanceCallback(prediction_model, valid_dataset, tp_obj.max_length)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=config.EPOCHS,
        callbacks=[edit_distance_callback],
        batch_size=config.BATCH_SIZE
    )
    return model


if __name__ == '__main__':
    model = train()
