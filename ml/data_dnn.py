
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf



# Order is important for the csv-readers, so we use an OrderedDict here.
defaults = collections.OrderedDict([
    ("datetime", [""]),
    ("month", [0]),
    ("day", [0]),
    ("hour", [0]),
    ("original", [0.0]),
    ("y", [0.0]),
    # hourly features
    ("y-10", [0.0]),
    ("y-9", [0.0]),
    ("y-8", [0.0]),
    ("y-7", [0.0]),
    ("y-6", [0.0]),
    ("y-5", [0.0]),
    ("y-4", [0.0]),
    ("y-3", [0.0]),
    ("y-2", [0.0]),
    ("y-1", [0.0]),
    #hourly deltas
    ("d-10", [0.0]),
    ("d-9", [0.0]),
    ("d-8", [0.0]),
    ("d-7", [0.0]),
    ("d-6", [0.0]),
    ("d-5", [0.0]),
    ("d-4", [0.0]),
    ("d-3", [0.0]),
    ("d-2", [0.0]),
    #daily features
    ("yy-5", [0.0]),
    ("yy-4", [0.0]),
    ("yy-3", [0.0]),
    ("yy-2", [0.0]),
    ("yy-1", [0.0]),
    ("yy-0", [0.0]),
    ("yyp1", [0.0]),
    ("yyp2", [0.0]),
    ("yyp3", [0.0]),
    ("yyp4", [0.0]),
    # deltas
    ("dd-5", [0.0]),
    ("dd-4", [0.0]),
    ("dd-3", [0.0]),
    ("dd-2", [0.0]),
    ("dd-1", [0.0]),
    ("dd-0", [0.0]),
    ("ddp1", [0.0]),
    ("ddp2", [0.0]),
    ("ddp3", [0.0]),
])  # pyformat: disable

Y_NAME = 'y'

types = collections.OrderedDict((key, type(value[0]))
                                for key, value in defaults.items())


def data_path():
    train = "./data/denver-features-train.csv"
    test = "./data/denver-features-test.csv"
    return train, test

def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    items = tf.decode_csv(line, list(defaults.values()))

    # Convert the keys and items to a dict.
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)

    # Remove the label from the features_dict
    label = features_dict.pop(Y_NAME)
    return features_dict, label

def has_no_question_marks(line):
    """Returns True if the line of text has no question marks."""
    # split the line into an array of characters
    chars = tf.string_split(line[tf.newaxis], "").values
    # for each character check if it is a question mark
    is_question = tf.equal(chars, "?")
    any_question = tf.reduce_any(is_question)
    no_question = ~any_question
    return no_question

def dataset():
    train_path, test_path = data_path()

    train = (tf.data
               .TextLineDataset(train_path)
               .skip(1)
               .filter(has_no_question_marks)
               .map(decode_line)
               .cache()
            )

    test = (tf.data
              .TextLineDataset(test_path)
              .skip(1)
              .filter(has_no_question_marks)
              .map(decode_line)
              .cache()
           )        

    return train, test