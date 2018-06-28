#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression using the DNNRegressor Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import data_dnn  

STEPS = 30000
VALUE_NORM_FACTOR = 50

def get_nn_model(features):
  model = tf.estimator.DNNRegressor(
      hidden_units=[32, 16, 16], 
      feature_columns=features,
      model_dir="./models/dnn",
  )
  return model

def get_linear_model(features):
  model = tf.estimator.LinearRegressor(
    feature_columns = features,
    model_dir = "./models/linear"
  )
  return model


def generate_features():
  #1. define the non-numeric features
  # The first way assigns a unique weight to each category. To do this you must
  # specify the category's vocabulary (values outside this specification will
  # receive a weight of zero). Here we specify the vocabulary using a list of
  # options. The vocabulary can also be specified with a vocabulary file (using
  # `categorical_column_with_vocabulary_file`). For features covering a
  # range of positive integers use `categorical_column_with_identity`.
  month_vocab = list(range(1,13))
  month = tf.feature_column.categorical_column_with_vocabulary_list(
    key = "month", vocabulary_list = month_vocab)

  day_vocab = list(range(1,32))
  day = tf.feature_column.categorical_column_with_vocabulary_list(
    key = "day", vocabulary_list = day_vocab)

  hour_vocab = list(range(0, 24))
  hour = tf.feature_column.categorical_column_with_vocabulary_list(
    key = "hour", vocabulary_list = hour_vocab)  

  #2. generate feature columns
  feature_columns = [
      tf.feature_column.numeric_column(key="y-1"),
      tf.feature_column.numeric_column(key="y-2"),
      tf.feature_column.numeric_column(key="y-3"),
      tf.feature_column.numeric_column(key="y-4"),
      tf.feature_column.numeric_column(key="y-5"),
      tf.feature_column.numeric_column(key="y-6"),
      tf.feature_column.numeric_column(key="y-7"),
      tf.feature_column.numeric_column(key="y-8"),
      tf.feature_column.numeric_column(key="y-9"),
      tf.feature_column.numeric_column(key="y-10"),

      # features of deltas
      # for linear models, these deltas seems do not help
      tf.feature_column.numeric_column(key="d-2"),
      tf.feature_column.numeric_column(key="d-3"),
      tf.feature_column.numeric_column(key="d-4"),
      tf.feature_column.numeric_column(key="d-5"),
      tf.feature_column.numeric_column(key="d-6"),
      tf.feature_column.numeric_column(key="d-7"),
      tf.feature_column.numeric_column(key="d-8"),
      tf.feature_column.numeric_column(key="d-9"),
      tf.feature_column.numeric_column(key="d-10"),

      # features of yesterday
      tf.feature_column.numeric_column(key="yy-5"),
      tf.feature_column.numeric_column(key="yy-4"),
      tf.feature_column.numeric_column(key="yy-3"),
      tf.feature_column.numeric_column(key="yy-2"),
      tf.feature_column.numeric_column(key="yy-1"),
      tf.feature_column.numeric_column(key="yy-0"),
      tf.feature_column.numeric_column(key="yyp1"),
      tf.feature_column.numeric_column(key="yyp2"),
      tf.feature_column.numeric_column(key="yyp3"),
      tf.feature_column.numeric_column(key="yyp4"),

      # deltas of yesterday
      # for linear models, these deltas seems do not help
      tf.feature_column.numeric_column(key="dd-5"),
      tf.feature_column.numeric_column(key="dd-4"),
      tf.feature_column.numeric_column(key="dd-3"),
      tf.feature_column.numeric_column(key="dd-2"),
      tf.feature_column.numeric_column(key="dd-1"),
      tf.feature_column.numeric_column(key="dd-0"),
      tf.feature_column.numeric_column(key="ddp1"),
      tf.feature_column.numeric_column(key="ddp2"),
      tf.feature_column.numeric_column(key="ddp3"),

      # Since this is a DNN model, convert categorical columns from sparse
      # to dense.
      # Wrap them in an `indicator_column` to create a
      # one-hot vector from the input.
      #tf.feature_column.indicator_column(month),
      # Or use an `embedding_column` to create a trainable vector for each
      # index.
      tf.feature_column.embedding_column(month, dimension=4),
      tf.feature_column.embedding_column(day, dimension=8),
      tf.feature_column.embedding_column(hour, dimension=8),
  ]

  return feature_columns

def main(argv):
  """Builds, trains, and evaluates the model."""
  assert len(argv) == 1
  print("tensorflow version: %s" % (tf.VERSION))
  (train, test) = data_dnn.dataset()

  # Build the training input_fn.
  def input_train():
    return (
        # Shuffling with a buffer larger than the data set ensures
        # that the examples are well mixed.
        train.shuffle(1000).batch(128)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next())

  
  feature_columns = generate_features()
  #model = get_linear_model(feature_columns)
  model = get_nn_model(feature_columns)

  # Train the model.
  model.train(input_fn=input_train, steps=STEPS)

  # Evaluate how the model performs on data it has not yet seen.
  full_eval(model)

  eval_model(model, test)
  return

def my_input_fun(dat):
  return (dat.batch(128).make_one_shot_iterator().get_next())

def full_eval(model):
  """calculate the RMSE on training and validation set."""
  train, test = data_dnn.dataset()

  r1 = model.evaluate(input_fn=lambda : my_input_fun(train))
  r2 = model.evaluate(input_fn=lambda : my_input_fun(test))

  loss1 = r1['average_loss']
  loss2 = r2['average_loss']

  rmse1 = VALUE_NORM_FACTOR * (math.sqrt(loss1))
  rmse2 = VALUE_NORM_FACTOR * (math.sqrt(loss2))
  print('\n' + 80 * "*")
  msg = "train RMSE: %.4f, test RMSE: %.4f" % (rmse1, rmse2)
  print("\n" + msg)
  return

def eval_model(model, test):
  """print the predicted values VS. real values."""
  values = model.predict(input_fn=lambda: my_input_fun(test))
  fname = "data/compare.dat"
  fout = open(fname, 'w')

  sess = tf.Session()
  yy = []
  nextv = test.make_one_shot_iterator().get_next() 
  while True:
    try:
      v = sess.run(nextv)
      yy.append(v[1])
    except tf.errors.OutOfRangeError:
      break  

  print("\n" + 80 * "*")
  pairs = zip(yy, values)
  line = "#y, y-hat\n"
  fout.write(line)
  for y, pred_dict in pairs:
      yh = pred_dict['predictions'][0]
      yh = yh * VALUE_NORM_FACTOR + 273.15
      y = y * VALUE_NORM_FACTOR + 273.15
      line = "%.4f, %.4f\n" % (y, yh)
      fout.write(line)

  fout.close()    
  return


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
