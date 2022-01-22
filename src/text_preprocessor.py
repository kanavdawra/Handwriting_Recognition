import tensorflow as tf


class TextPreprocessor:

    def __init__(self, dataframe=None):
        if dataframe is not None:
            self.set_vocabulary(dataframe)
            self.set_max_label_length(dataframe)
            self.set_char_to_num()
            self.set_num_to_char()
        else:
            print("Pass Dataframe or list of Strings!!!")

    def set_vocabulary(self, dataframe):
        self.characters = set(char for label in dataframe for char in label)

    def set_max_label_length(self, dataframe):
        self.max_length = max([len(label) for label in dataframe])

    def set_char_to_num(self):
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=list(self.characters), mask_token=None)

    def set_num_to_char(self):
        self.num_to_char = tf.keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(),
                                                        mask_token=None, invert=True)