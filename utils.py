"""
Utility functions for handling tokens, intensities, etc.

TensorFlow and Midi-related libraries not needed here.
"""

import numpy as np
import os
import random


def split_token_to_arrays(tokens, split_number):
    """
    Given a token list, `tokens`, turns the list into a list of lists of tokens using `split_number` as the token to split on.
    
    Returns the tokens as a list of lists of tokens.
    """

    output = []
    output.append([])
    split_number = str(split_number)
    for token in tokens:
        if token != split_number:
            output[-1].append(token)
        else:
            output.append([])
    
    return output


def join_token_array(tokens, join_number):
    """
    Opposite of `split_token_to_arrays`. Joins a token list into a list with splits replaced with `join_number`.

    Returns the joined list.
    """

    output = []
    for token in tokens:
        output += token
        output += [join_number]
    
    return output


def remove_duplicates(tokens):
    """
    Removes duplicates from a list of lists of tokens.

    Returns the token array without duplicates per token list.
    """

    output = []
    for array in tokens:
        new_array = []

        for value in array:
            if not (value in new_array):
                new_array.append(value)

        output.append(new_array)
    
    return output


def get_tokens_between_beats(start_beat, end_beat, tokens):
    """
    Given a list of lists of tokens (`tokens`) and a start beat and end beat,
    calculates the tokens between the two beats.

    Returns both the tokens between the two beat, as well as how many tokens are between the beats.
    """

    indexes_to_take = end_beat - start_beat

    this_tokens = []
    this_tokens_total = 0

    # print(start_beat + indexes_to_take)
    # print(len(tokens))

    if start_beat + indexes_to_take > len(tokens):
        return [], 0
    
    for index in range(indexes_to_take):
        # print(len(tokens))

        this_tokens.append(tokens[start_beat + index])
        this_tokens_total += len(tokens[start_beat + index])
        # print(tokens[this_beat + index])
    
    return this_tokens, this_tokens_total


def combine_tokens(tokens):
    """
    Combines an list of token lists into a single token list.

    Returns the combined list.
    """
    
    max_length = -1
    for item in tokens:
        if len(item) > max_length:
            max_length = len(item)
    
    combined = []
    for index in range(max_length):
        combined.append([])
        for row in tokens:
            if index >= len(row):
                continue
            combined[-1] += (row[index])

    output = combined
    return output


def tokens_are_empty(tokens):
    """
    Check if a token array/list is actually empty.

    i.e., `[[], [], []]` is technically not empty, but for our purposes, it is.

    Returns whether it is empty.
    """

    if len(tokens) == 0:
        return True
    
    for token in tokens:
        if token != 0:
            return False
    
    return True


def get_random_cut(inputs, length):
    """
    Gets a random cut, or 'window', from `inputs` of length `length`.

    Returns the cut as a list of length `length`.
    """

    selected = random.choice(inputs)
    start = int(np.random.uniform(0, len(selected) - length))

    output = []
    for item in range(length):
        output.append(selected[start + item])
    
    return output


def pad_input(input, buffer_size):
    """
    Pads an input formatted as a list of token lists, `input`, to the given size, `buffer_size`.

    Padded with empty lists, `[]`, at the front.

    Returns the padded input.
    """

    if len(input) < (buffer_size - 1):
        print("Padding: " + str(input))
        input = [[]] * ((buffer_size - 1) - len(input)) + input
        print("Padded to: " + str(input))

    return input


def get_pitches_from_output(prediction, dictionary, cutoff=0.5, num_to_force=0):
    """
    Get pitches from the model's output (`prediction`), converting to pitches using `dictionary`.
    `cutoff` determines how activate a unit has to be (between `0` and `1`) to be considered a valid output.

    Optionally can force `num_to_force` outputs to be valid. The most active are forced first.

    Returns the output pitches as a list.
    """

    output = []
    prediction = prediction[0]

    # handle cutoff
    for idx in range(len(prediction)):
        confidence = prediction[idx]
        if confidence >= cutoff:
            pitch = next(pitch for pitch, value in dictionary.items() if value == idx)
            output.append(str(pitch))
    
    # handle forced
    forced = 0
    while (forced != num_to_force) and (len(prediction) >= 0):
        max_value = np.argmax(prediction)
        prediction = np.delete(prediction, max_value)
        pitch = next(pitch for pitch, value in dictionary.items() if value == max_value)
        forced += 1
        if str(pitch) in output:
            continue
        output.append(str(pitch))

    return output


def generate_random_input(buffer_size, token_count):
    """
    Generates a random input.
    (This will likely not yield very good results...)
    """

    random_inputs = []
    for x in range(buffer_size - 1):
        generated = np.random.randint(1, token_count, np.random.randint(1, 5, 1)).tolist()
        generated = list(map(str, generated))
        random_inputs.append(generated)
    
    return random_inputs

def generate_basic_drums(buffer_size, token_count):
    """
    Generates a basic drum pattern as an input.

    1. Bass drum
    2. Hi-hat
    3. Snare
    4. Hi-hat
    5. Repeat...
    """
    programmed_input = []

    drums_length = int((buffer_size - 1) // 4) * 4

    for x in range(drums_length):
        programmed_input.append(["i0"])

    # force drums on beats 1, 3, and snare on 2, 4, hh on off beats
    for x in range(len(programmed_input)):
        if (x % 4) == 0: # kick
            programmed_input[x].append("35")
        if ((x + 2) % 4) == 0: # snare
            programmed_input[x].append("38")
        programmed_input[x].append("42") # hh
    
    programmed_input = pad_input(programmed_input, buffer_size)

    return programmed_input


def generate_blank_input(buffer_size, token_count):
    """
    Generates a completely blank input.
    """

    return [[]]


def generate_count_in(buffer_size, token_count):
    """
    Generates a count in of four hi-hats.
    """

    programmed_input = [["42"], [], ["42"], [], ["42"], [], ["42"], []]

    for i in programmed_input:
        i.append("i-2")

    return programmed_input


def categorical_encoder(tokens, num_classes, dictionary):
    """
    A categorical encoder function to turn `tokens` into categorically encoded classes of
    total `num_classes` using `dictionary` to look up class encodings.

    Returns the tokens categorically encoded as a `numpy` array.
    """

    array = np.zeros(shape=(len(tokens), num_classes))

    for token_array_idx in range(len(tokens)):
        token_array = tokens[token_array_idx]

        for token_idx in range(len(token_array)):
            token = token_array[token_idx]

            if "i" in token:
                array[token_array_idx][dictionary[-999]] = float(token.replace("i", ""))
            else:
                token = int(token)

                if token in dictionary:
                    array[token_array_idx][dictionary[token]] = 1

    return array


def input_to_categories(input, token_count, buffer_size, dictionary):
    """
    Converts an input (list of tokens) into a categories for inputting into the neural network.

    Returns the input in category encoding as a valid network input.
    """

    input = pad_input(input, buffer_size)
    new_input = categorical_encoder(input, token_count, dictionary)
    new_input = new_input.reshape(-1, buffer_size - 1, token_count)

    return new_input


def intensities_to_float(tokens):
    """
    Converts intensity tokens as a list of strings into a list of floats.
    Ignores non-intensity tokens.

    Returns the list of floats.
    """

    intensities = []
    for beat in tokens:
        for token in beat:
            if "i" in token:
                intensities.append(float(token.replace("i", "")))
    
    return intensities


def extract_intensities(tokens):
    """
    Extracts only the intensities from a given token list.

    Returns just the intensities as a list seperated by timestep.
    """

    intensities = []
    for beat in tokens:
        for token in beat:
            if "i" in token:
                intensities.append([token])
    
    return intensities


def remove_intensity_outliers(intensity_array, mean, std):
    """
    Removes intensity outliers by replacing them with the closest non-outlier intensity.

    Outliers are defined by being either bigger or smaller than `mean +/- (2 * std)`.

    Returns a list of intensities with outliers removed.
    """

    new_array = []
    for intensity in intensity_array:
        if intensity < mean + (2 * std):
            if intensity > mean - (2 * std):
                new_array.append(intensity)
                continue
        new_array.append(-999)
    
    newer_array = []
    
    for index in range(len(new_array)):
        intensity = new_array[index]
        if intensity != -999:
            newer_array.append(intensity)
            continue

        prev_actual = -999
        search_index = index
        while (prev_actual == -999 and search_index >= 0):
            this_intensity = new_array[search_index]
            if this_intensity != -999:
                prev_actual = this_intensity
                break
            search_index -= 1
        
        next_actual = -999
        search_index = index
        while (next_actual == -999 and search_index < len(new_array)):
            this_intensity = new_array[search_index]
            if this_intensity != -999:
                next_actual = this_intensity
                break
            search_index += 1
        
        if prev_actual == -999:
            prev_actual = next_actual
        if next_actual == -999:
            next_actual = prev_actual

        average = np.mean([prev_actual, next_actual])
        newer_array.append(average)

    return newer_array


def normalize_intensities(intensity_tokens):
    """
    Normalize intensities by removing outliers and then subtracting the mean and dividing by the standard deviation.

    Returns normalized intensities.
    """

    unpacked = []
    for beat in intensity_tokens:
        intensity = float(beat[0].replace("i", ""))
        unpacked.append(intensity)
    
    mean = np.mean(unpacked)
    std = np.std(unpacked)

    intensity_tokens = remove_intensity_outliers(unpacked, mean, std)

    # recalc without extremes
    mean = np.mean(intensity_tokens)
    std = np.std(intensity_tokens)

    new_array = []
    for beat in intensity_tokens:
        intensity = beat
        intensity -= mean
        intensity /= std
        new_array.append(["i" + str(round(intensity, 1))])
    
    return new_array


def clean_temp():
    """
    Clean up temporary files.
    """

    if os.path.exists("temp.mid"):
        os.remove("temp.mid")