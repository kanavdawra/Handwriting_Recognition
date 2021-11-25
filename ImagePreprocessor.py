import tensorflow as tf
import config


class ImagePreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def read_image(path):
        return tf.io.read_file(path)

    @staticmethod
    def decode_image(image, image_type):
        if image_type == "jpg" or "jpeg":
            return tf.io.decode_jpeg(image, channels=config.IMAGE_CHANNELS)
        elif image_type == "png":
            return tf.io.decode_png(image, channels=config.IMAGE_CHANNELS)
        else:
            print("Image type must me jpg, jpeg, png")

    @staticmethod
    def convert_image_float32(image):
        return tf.image.convert_image_dtype(image, tf.float32)

    @staticmethod
    def image_resize(image, resize=None):
        if resize is None:
            h = config.IMAGE_HEIGHT
            w = config.IMAGE_WIDTH
        else:
            h = resize[1]
            w = resize[0]
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        # Check tha amount of padding needed to be done.
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )
        print(image.shape)
        return image

    @staticmethod
    def transpose_image(image, perm=None):
        if perm is not None:
            return tf.transpose(image, perm=perm)

        else:
            print("'perm' cannot be None, pass values such as `[1, 0, 2]`")

    @staticmethod
    def flip_left_to_right(image):
        return tf.image.flip_left_right(image)

