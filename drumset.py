"""
Contains the `DrumSetSequence` class.

Uses TensorFlow, Keras and numpy
"""

from tensorflow import keras
import numpy as np
import utils
import os


class DrumSetSequence(keras.utils.Sequence):
    """
    This class extends `keras.utils.Sequence` to wrap the drumset pre-processed data in an easy-to-access (for the neural network) way.
    """

    def __init__(self, inputs, batch_size, buffer_size, token_count, dictionary):
        self.batch_size = batch_size
        self.inputs = inputs
        self.buffer_size = buffer_size
        self.token_count = token_count
        self.dictionary = dictionary


    def __getitem__(self, index):
        """
        Used by TensorFlow when training to fetch inputs and target outputs.
        Returns two values, the inputs `Xs` (previous timesteps and intensities) and the output `ys` (next timestep, no intensity)

        `index` is not used as the fetched window is random.
        """

        Xs = np.zeros((self.buffer_size - 1, self.token_count, self.batch_size))
        ys = np.zeros((self.token_count, self.batch_size))

        for x in range(self.batch_size):
            cut = utils.get_random_cut(self.inputs, self.buffer_size)

            new_input = utils.categorical_encoder(cut, self.token_count, self.dictionary)

            output_x = new_input[:self.buffer_size - 1]
            output_y = new_input[self.buffer_size - 1:]

            output_x = output_x.astype("float32")
            output_y = output_y.astype("float32")

            Xs[:,:,x] = output_x
            ys[:,x] = output_y

        # delete final column as intensity not needed as prediction.
        ys = np.delete(ys, [self.token_count - 1], 0)
        
        Xs = Xs.transpose((2, 0, 1))
        ys = ys.transpose((1, 0))
        
        return Xs, ys
    
    
    def __len__(self):
        """
        Not really applicable for this style of dataset.

        Returns an appropriate value given the batch size.
        """

        return int(len(self.inputs) / self.batch_size)


    def on_epoch_end(self):
        pass

