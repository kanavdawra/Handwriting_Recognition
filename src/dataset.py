import tensorflow as tf
from src import text_preprocessor as tp
from src import image_preprocessor as ip
from src import config


class ClassificationDataset:
    def __init__(self, labels):
        self.tp_obj = tp.TextPreprocessor(labels)
        self.ip_obj = ip.ImagePreprocessor()
        self.char_to_num = self.tp_obj.char_to_num
        self.max_len = self.tp_obj.max_length

    def vectorize_label(self, label):
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=config.PADDING_TOKEN)
        return label

    def get_data_object(self, path, label):
        image = self.ip_obj.read_image(path)
        image = self.ip_obj.decode_image(image, "jpg")
        image = self.ip_obj.convert_image_float32(image)
        image = self.ip_obj.image_resize(image=image)
        image = self.ip_obj.transpose_image(image, [1, 0, 2])
        image = self.ip_obj.flip_left_to_right(image)
        label = self.vectorize_label(label)
        return {"image": image, "label": label}

    def get_dataset(self, image_paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = (dataset.map(self.get_data_object, num_parallel_calls=tf.data.AUTOTUNE)) \
            .batch(config.BATCH_SIZE) \
            .cache() \
            .prefetch(buffer_size=tf.data.AUTOTUNE) \

        return dataset
