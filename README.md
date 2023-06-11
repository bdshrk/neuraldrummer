# NeuralDrummer

**A neural network for generating drum tracks for songs.**

Practical project for the COMP6590: Computational Creativity module.

## Usage

The code can be run from the Jupyter notebook file `main.ipynb`.

You will need `TensorFlow` installed in your Python environment and
also `PrettyMIDI`, `Mido`, `numpy`, and `matplotlib`. The following command should
work:

```
pip install pretty_midi mido numpy tensorflow matplotlib
```

Each cell in the notebook should be executed consecutively with the
exception of the `nn.train()` and `nn.plot()` cells (which are for training
the network if you wish.) The model saves its weights to the
`/saved/` directory and can be loaded in the cell `nn.load()`.

Feel free to modify the `INPUT_PATH` in the final cell to point to a MIDI
file of your choosing. You can also modify the cut-off parameter of the
`tokeniser.add_drum_track()` within the same cell to adjust the sensitivity
of the result. You should find the output as a file named `combined.mid`.

In order for the model to learn, you will require a collection of MIDI files
containing drum tracks.
During development, I used the *"Lakh MIDI Dataset Clean"*, available
[here](https://colinraffel.com/projects/lmd/). Once the MIDI files have been
pre-processed, the original files are no longer needed. The result of the 
pre-processing is stored in a file named `saved.txt`.

Note: You will need a fair amount of memory to load the neural network and
the inputs from the saved file.
