"""
Contains the `Model` class.

Uses TensorFlow, Keras and MatPlotLib
"""

from tensorflow import keras
from keras import layers, callbacks
from matplotlib import pyplot as plt
import os
import utils


class Model():
    """
    This class is a wrapper for TensorFlow/Keras functions. It provides functionality for creating, training, saving, and loading neural network models.
    """

    def __init__(self, dataset, buffer_size, token_count, batch_size, dictionary):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.token_count = token_count
        self.batch_size = batch_size
        self.dictionary = dictionary
        self.history = None
        self.callbacks_array = []

        self.create_new_model()
    

    def create_new_model(self):
        """
        Creates a model using the hard-coded architecture.
        """

        self.model = keras.Sequential()

        # input is minus all negative tokens
        NETWORK_TOKEN_INPUT_COUNT = self.token_count
        NETWORK_TOKEN_OUTPUT_COUNT = self.token_count - 1

        print("Creating model with " + str(self.buffer_size - 1) + " x " + str(NETWORK_TOKEN_INPUT_COUNT) + " inputs and 1 x " + str(NETWORK_TOKEN_OUTPUT_COUNT) + " outputs.")

        self.model.add(layers.Input(shape=(self.buffer_size - 1,NETWORK_TOKEN_INPUT_COUNT, )))

        self.model.add(layers.LSTM(units=512, return_sequences=True))
        self.model.add(layers.LSTM(units=512, return_sequences=True))
        self.model.add(layers.LSTM(units=512))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Dense(256))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(128))
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Dense(units=NETWORK_TOKEN_OUTPUT_COUNT, activation="sigmoid"))

        self.model.summary()
        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001))

        # checkpoints
        checkpoint_filepath = "./saved/saved"
        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="loss",
            mode="auto",
            save_best_only=False)

        self.callbacks_array = [
            model_checkpoint_callback
        ]
    

    def train(self, epochs):
        """
        Train the model for X epochs.
        """

        self.history = self.model.fit(self.dataset, epochs=epochs, callbacks=self.callbacks_array)


    def save(self):
        """
        Saves the model weights to `./saved/`
        """

        self.model.save_weights("./saved/saved")


    def load(self):
        """
        Loads the model weights from files at `./saved/`
        """

        if os.path.exists("./saved/saved.index"):
            self.model.load_weights("./saved/saved")
    

    def plot(self):
        """
        Plot model loss history
        """

        if self.history != None:
            plt.plot(self.history.history["loss"])
            plt.grid()
            plt.show()
        else:
            print("No history to show.")

    
    def predict(self, input):
        """
        Predict a single timestep using the given input.
        """

        return self.model.predict(input, verbose=0)
    

