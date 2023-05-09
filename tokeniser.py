"""
Contains functionality for converting MIDI files into tokens and back.

Most functions make use of either PrettyMidi or Mido.
"""

import pretty_midi as pm
import mido
from mido import MidiFile, MidiTrack
from mido import Message, MetaMessage

import numpy as np
import utils
import os


def instrument_to_token(beats, instrument):
    """
    Converts a single `PrettyMidi` instrument into a token array with the array split by values in `beats`.

    Returns the token array.
    """

    new_beats = beats
    last_index = 0
    output = []

    notes = instrument.notes

    notes.sort(key=lambda x : x.start)

    np_beats = np.array(new_beats)

    for note in notes:
        beat_index = np.abs(np_beats - note.start).argmin()

        # handle new beats
        difference = max(beat_index - last_index, 0)
        if difference > 0:
            for empty in range(difference):
                output.append("-1")
        
        last_index = beat_index
        output.append(str(note.pitch))
    
    output = utils.split_token_to_arrays(output, -1)

    return output


def get_beats(mid):
    """
    Extracts the beats of a `PrettyMidi` midi.

    Returns two values, an array of the beats, and the time between the beats (`delta_beats`).
    """

    beats = mid.get_beats()
    delta_beats = []
    new_beats = []

    for beat_idx in range(0, len(beats) - 1):
        beat = beats[beat_idx]
        delta = beats[beat_idx + 1] - beat
        delta_beats.append(delta / 2)
        delta_beats.append(delta / 2)
        new_beats.append(beat)
        new_beats.append(beat + (delta * 2/4))

    # add last beats
    buffer_beats = 32
    for x in range(buffer_beats):
        new_beats.append(new_beats[-1] + delta_beats[-1])
        delta_beats += [delta_beats[-1]]

    return new_beats, delta_beats


def get_beat_intensities(beats, tokens):
    """
    Get the beat intensities given a list of beats and a corresponding list of tokens.
    Intensity is defined as the number of tokens that occur within a beat's duration.

    Returns a list of the intensities per beat.
    """

    intensities = []

    for beat_index in range(len(beats) - 1):
        this_beat = beats[beat_index]
        next_beat = beats[beat_index + 1]
        
        this_tokens, this_tokens_total = utils.get_tokens_between_beats(this_beat, next_beat, tokens)

        intensities.append([this_beat, this_tokens_total])
    
    return intensities


def get_downbeat_indexes(mid, beats):
    """
    From a `PrettyMidi` file and associated beats array, finds the downbeats (starting beats of a bar) and
    returns the indexes of these beats in the provided beats array.

    This takes into account changes in tempo, time signatures, etc...

    Returns a list of the indexes of downbeats.
    """

    downbeats = mid.get_downbeats()

    # handle downbeats that extend final bar...
    downbeats = list(downbeats)
    last_delta = downbeats[-1] - downbeats[-2]
    downbeats.append(downbeats[-1] + last_delta)
    downbeats.append(downbeats[-1] + last_delta)
    downbeats.append(downbeats[-1] + last_delta)
    downbeats.append(downbeats[-1] + last_delta)

    indexes = []

    np_beats = np.array(beats)
    for downbeat in downbeats:
        beat_index = np.abs(np_beats - downbeat).argmin()
        if not (beat_index in indexes):
            indexes.append(beat_index)
    
    return indexes


def intensities_to_tokens(intensities, tokens):
    """
    Convert intensities to a token array for merging with other token arrays.

    Intensities should occur one beat before the actual beat so the network
    knows the next beat's intensity and can generate accordingly.

    Returns a list of intensity tokens.
    """
    array = []

    for index in range(len(intensities) - 1):
        this_beat = intensities[index][0]
        next_beat = intensities[index + 1][0]
        intensity = "i" + str(intensities[index][1])

        between_tokens, total = utils.get_tokens_between_beats(this_beat, next_beat, tokens)

        for token_array_idx in range(len(between_tokens)):
            array.append([intensity])
    
    array = np.roll(array, -1)
    array[-1] = array[-2]
    array = array.tolist()

    # handle beyond final bar
    for x in range(4):
        array.append(array[-1])

    return array


def file_to_token(path, return_instruments=False):
    """
    Turns a MIDI file into tokens.
    If `return_instruments`, it will consider and return tokens from ALL instruments, otherwise, just drums/percussion.

    Returns `None` if failed to extract tokens, otherwise, returns the tokens.
    """

    # sometimes prettymidi will fail to open the midi (file corrupt, empty, etc...)
    mid = None
    try:
        mid = pm.PrettyMIDI(path)
    except:
        print("Broken!")
        return None

    beats, deltas = get_beats(mid)
    downbeats = get_downbeat_indexes(mid, beats)
    tokens = []
    drum_track = []

    # tokenise all instruments to find intensities...
    for instrument in mid.instruments:
        this_token = instrument_to_token(beats, instrument)

        tokens.append(this_token)
    
    tokens = utils.combine_tokens(tokens)

    # ignore short files that wont contain useful info (specifically about drum tracks)
    if len(tokens) <= 32:
        print("File too short!")
        return None

    beats_as_list = list(range(0, int(downbeats[-1]) + 1))
    beat_intensities = get_beat_intensities(beats_as_list, tokens)
    intensities_tokens = intensities_to_tokens(beat_intensities, tokens)

    if len(intensities_tokens) == 0:
        print("Failed to find beats!")
        return None

    normalized_intensity = utils.normalize_intensities(intensities_tokens)

    if return_instruments:
        tokens = utils.combine_tokens([tokens, normalized_intensity])
        return tokens

    # find drum track and tokenise
    drum_track = []
    for instrument in mid.instruments:
        if not instrument.is_drum:
            continue
        
        drum_track = instrument_to_token(beats, instrument)

    if utils.tokens_are_empty(drum_track):
        print("Empty!")
        return None

    drum_track = utils.combine_tokens([drum_track, normalized_intensity])
    return drum_track


def tokenize_drums(process_limit, base_dir):
    """
    Processes any MIDI files in the given directory, `base_dir`, turning them into token strings stored in `saved.txt`.
    The process is recursive and searches all folders within `base_dir` as well.

    Set `process_limit` to `-1` to process all possible files.
    """

    completed = 0
    inputs = []

    for dir_path, dir_names, file_names in os.walk(base_dir):
        for file_name in file_names:
            file = os.path.join(dir_path, file_name)
            if not file.endswith(".mid"):
                continue

            print("Processing: " + str(file_name))

            output = file_to_token(os.path.join(base_dir, file))

            if output != None:
                inputs.append(utils.join_token_array(output, "-1"))
                completed += 1
            
            if process_limit != -1 and completed >= process_limit:
                break
            
        if process_limit != -1 and completed >= process_limit:
            break

    print("Found " + str(len(inputs)) + " valid inputs!")

    # write the file
    disk_file = open("saved.txt", "wt")
    for line in inputs:
        disk_file.write(",".join(map(str, line)) + "\n")
    disk_file.close()


def tokens_to_track(tokens, ticks_per_beat):
    """
    Converts a list of tokens into a `MidiTrack` object from `Mido`.
    This should only be used for drum tracks as Midi channel 9 is hard-coded (channel 9, or 10 starting from 1, is reserved for percussion as per General MIDI specification.)

    Returns the `MidiTrack`.
    """

    ticks_per_beat = int(ticks_per_beat / 2)
    velocity = 127

    track = MidiTrack()
    track.append(Message("program_change", program=0, channel=9, time=0))

    for step in tokens:
        track.append(Message("note_on", channel=9, note=0, velocity=velocity, time=0))

        tokens_this_step = []
        for pitch in step:
            # dont process intensities or invalid pitches...
            if "i" in pitch:
                continue
            if int(pitch) <= 0:
                continue

            track.append(Message("note_on", channel=9, note=int(pitch), velocity=velocity, time=0))
            tokens_this_step.append(int(pitch))

        track.append(Message("note_off", channel=9, note=0, velocity=velocity, time=ticks_per_beat))

        for token in tokens_this_step:
            track.append(Message("note_off", channel=9, note=token, velocity=velocity, time=0))

    # stop playback cutting off too soon at end of file.
    track.append(Message("note_on", channel=9, note=0, velocity=velocity, time=0))
    track.append(Message("note_off", channel=9, note=0, velocity=velocity, time=ticks_per_beat * 4))
    track.append(MetaMessage("end_of_track", time=0))

    return track


def tokens_to_midi(tokens, tempo):
    """
    Converts tokens into a `.mid` Midi file that plays at a given tempo using `Mido` functions.
    """

    mid = MidiFile(ticks_per_beat=320, type=1)
    mid.tracks.append(tokens_to_track(tokens, 320))

    # meta track
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    track.append(MetaMessage("end_of_track", time=0))
    
    mid.save("new_song.mid")


def remove_drum_track(file_name):
    """
    Uses `PrettyMidi` to remove all drum tracks from a given input.

    Saves the resulting file to `temp.mid`.
    """

    mid = pm.PrettyMIDI(file_name)

    new_instruments = []
    for instrument in mid.instruments:
        if not instrument.is_drum:
            new_instruments.append(instrument)
    
    mid.instruments = new_instruments
    
    mid.write("temp.mid")


def append_drum_track(file_name, tokens):
    """
    Uses `Mido` to add a list of tokens to the given input Midi.
    The input Midi is not modified and the resulting Midi to saved as `combined.mid`.
    """

    mid = MidiFile(file_name)

    mid.tracks.append(tokens_to_track(tokens, mid.ticks_per_beat))

    mid.save("combined.mid")


def add_drum_track(model, buffer_size, token_count, dictionary, midi_path, starting_input, cutoff):
    """
    Takes a model and associated parameters and generates a drum track using the model's `predict` function.

    Specify the Midi file to calculate characteristics and metrics from using `midi_path`.
    Models take a starting input that can be specified by `starting_input` and using one of the generator functions.
    Takes a cut-off, `cutoff`, for which output activations under this value are ignored.

    Returns a list of tokens containing the output predicted drum track.
    """

    instruments = file_to_token(midi_path, True)
    length = len(instruments)
    print("Length: " + str(length))

    current_input = utils.pad_input(starting_input, buffer_size)

    # handle intensity calc
    intensities = utils.extract_intensities(instruments)
    intensities = utils.normalize_intensities(intensities)
    intensities = utils.intensities_to_float(intensities)

    output = []
    for index in range(length):
        test_input = utils.input_to_categories(current_input, token_count, buffer_size, dictionary)

        for buffer_index in range(buffer_size - 1):
            token_index = index - buffer_index - 1
            if token_index < 0:
                continue

            test_input[:, (buffer_size - 2) - buffer_index, (token_count - 1)] = intensities[token_index]

        print("Latest input of " + str(test_input[:, -1]))
        print("Full input: \n" + str(test_input))
        prediction = model.predict(test_input)

        values = utils.get_pitches_from_output(prediction, dictionary, cutoff, 1)
        new_input = []

        # shift input along
        for index_2 in range(1, len(current_input)):
            new_input.append(current_input[index_2])
        new_input.append(values)

        current_input = new_input
        output.append(values)
        print("Index " + str(index)  + " : " + str(values))
    
    return output


def load_inputs_from_file(file_path, buffer_size):
    """
    Load pre-processed inputs from a saved file on disk.
    Will only load saved inputs that are longer than the `buffer_size`.
    
    Returns the inputs as a list of token lists.
    """

    inputs = []

    if not os.path.exists(file_path):
        print("No saved file...")
        return
    
    disk_file = open(file_path, "rt")
    data = disk_file.read().split("\n")
    for line in data:
        split = line.split(",")

        # skip empty new line at end of file
        if len(split) <= 1:
            continue

        if len(split) > buffer_size:
            inputs.append(utils.split_token_to_arrays(split, -1))
    
    return inputs
