import numpy as np
import time
import argparse
import json
from math import sqrt, ceil
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load data set
def _load_dataset(file_path, header_row_index, index_col_name, col_to_predict, cols_to_drop):
  """
    file_path: the csv file path
    header_row_index: the header row index in the csv file
    index_col_name: the index column (can be None if no index is there)
    col_to_predict: the column name/index to predict
    cols_to_drop: the column names/indices to drop (single label or list-like)
  """

  # read dataset from disk
  dataset = read_csv(file_path, header=header_row_index, index_col=False)

  # set index col
  if index_col_name:
    dataset.set_index(index_col_name, inplace=True)

  # drop nonused colums
  if cols_to_drop:
    if type(cols_to_drop[0]) == int:
      dataset.drop(index=cols_to_drop, axis=0, inplace=True)
    else:
      dataset.drop(columns=cols_to_drop, axis=1, inplace=True)

  # get rows and column names
  col_names = dataset.columns.values.tolist()
  values = dataset.values

  # move the column to predict to be the first col:
  col_to_predict_index = col_to_predict if type(col_to_predict) == int else col_names.index(col_to_predict)
  output_col_name = col_names[col_to_predict_index]
  if col_to_predict_index > 0:
    col_names = [col_names[col_to_predict_index]] + col_names[:col_to_predict_index] + col_names[col_to_predict_index+1:]
  values = np.concatenate((values[:, col_to_predict_index].reshape((values.shape[0], 1)), values[:,:col_to_predict_index], values[:,col_to_predict_index+1:]), axis=1)

  # ensure all data is float
  values = values.astype("float32")

  return (col_names, values, values.shape[1], output_col_name)

# scale dataset
def _scale_dataset(values, scale_range):
  """
    values: dataset values
    scale_range: scale range to fit data in
  """

  # normalize features
  scaler = MinMaxScaler(feature_range=scale_range or (0, 1))
  scaled = scaler.fit_transform(values)

  return (scaler, scaled)

# convert series to supervised learning (ex: var1(t)_row1 = var1(t-1)_row2)
def _series_to_supervised(values, n_in, n_out, dropnan, col_names, verbose):
  """
    values: dataset scaled values
    n_in: number of time lags (intervals) to use in each neuron
    n_out: number of time-steps in future to predict
    dropnan: whether to drop rows with NaN values after conversion to supervised learning
    col_names: name of columns for dataset
    verbose: whether to output some debug data
  """

  n_vars = 1 if type(values) is list else values.shape[1]
  if col_names is None: col_names = ["var%d" % (j+1) for j in range(n_vars)]
  df = DataFrame(values)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
    else:
      names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]

  # put it all together
  agg = concat(cols, axis=1)
  agg.columns = names

  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)

  if verbose:
    print("\nsupervised data shape:", agg.shape)

  return agg

# split into train and test sets
def _split_data_to_train_test_sets(values, n_intervals, n_features, train_percentage, verbose):
  """
    values: dataset supervised values
    n_intervals: number of time lags (intervals) to use in each neuron
    n_features: number of features (variables) per neuron
    train_percentage: percentage of train data related to the dataset series size; (1-train_percentage) will be for test data
    verbose: whether to output some debug data
  """

  n_train_intervals = ceil(values.shape[0] * train_percentage)
  train = values[:n_train_intervals, :]
  test = values[n_train_intervals:, :]

  # split into input and outputs
  n_obs = n_intervals * n_features
  train_X, train_y = train[:, :n_obs], train[:, -n_features]
  test_X, test_y = test[:, :n_obs], test[:, -n_features]

  # reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], n_intervals, n_features))
  test_X = test_X.reshape((test_X.shape[0], n_intervals, n_features))

  if verbose:
    print("")
    print("train_X shape:", train_X.shape)
    print("train_y shape:", train_y.shape)
    print("test_X shape:", test_X.shape)
    print("test_y shape:", test_y.shape)

  return (train_X, train_y, test_X, test_y)

# create the nn model
def _create_model(train_X, train_y, test_X, test_y, n_neurons, n_batch, n_epochs, is_stateful, has_memory_stack, loss_function, optimizer_function, draw_loss_plot, output_col_name, verbose):
  """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    n_neurons: number of neurons for LSTM nn
    n_batch: nn batch size
    n_epochs: training epochs
    is_stateful: whether the model has memory states
    has_memory_stack: whether the model has memory stack
    loss_function: the model loss function evaluator
    optimizer_function: the loss optimizer function
    draw_loss_plot: whether to draw the loss history plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
  """

  # design network
  model = Sequential()

  if is_stateful:
    # calculate new compatible batch size
    for i in range(n_batch, 0, -1):
      if train_X.shape[0] % i == 0 and test_X.shape[0] % i == 0:
        if verbose and i != n_batch:
          print ("\n*In stateful network, batch size should be dividable by training and test sets; had to decrease it to %d." % i)
        n_batch = i
        break

    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True, return_sequences=has_memory_stack))
    if has_memory_stack:
      model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True))
  else:
    model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))

  model.add(Dense(1))

  model.compile(loss=loss_function, optimizer=optimizer_function)

  if verbose:
    print("")

  # fit network
  losses = []
  val_losses = []
  if is_stateful:
    for i in range(n_epochs):
      history = model.fit(train_X, train_y, epochs=1, batch_size=n_batch, validation_data=(test_X, test_y), verbose=0, shuffle=False)

      if verbose:
        print("Epoch %d/%d" % (i + 1, n_epochs))
        print("loss: %f - val_loss: %f" % (history.history["loss"][0], history.history["val_loss"][0]))

      losses.append(history.history["loss"][0])
      val_losses.append(history.history["val_loss"][0])

      model.reset_states()
  else:
    history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch, validation_data=(test_X, test_y), verbose=2 if verbose else 0, shuffle=False)

  if draw_loss_plot:
    pyplot.plot(history.history["loss"] if not is_stateful else losses, label="Train Loss (%s)" % output_col_name)
    pyplot.plot(history.history["val_loss"] if not is_stateful else val_losses, label="Test Loss (%s)" % output_col_name)
    pyplot.legend()
    pyplot.show()

  return (model, n_batch)

# make a prediction
def _make_prediction(model, train_X, train_y, test_X, test_y, compatible_n_batch, n_intervals, n_features, scaler, draw_prediction_fit_plot, output_col_name, verbose):
  """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    compatible_n_batch: modified (compatible) nn batch size
    n_intervals: number of time lags (intervals) to use in each neuron
    n_features: number of features (variables) per neuron
    scaler: the scaler object used to invert transformation to real scale
    draw_prediction_fit_plot: whether to draw the the predicted vs actual fit plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
  """

  if verbose:
    print("")

  yhat = model.predict(test_X, batch_size=compatible_n_batch, verbose = 1 if verbose else 0)
  test_X = test_X.reshape((test_X.shape[0], n_intervals*n_features))

  # invert scaling for forecast
  inv_yhat = np.concatenate((yhat, test_X[:, (1-n_features):]), axis=1)
  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,0]

  # invert scaling for actual
  test_y = test_y.reshape((len(test_y), 1))
  inv_y = np.concatenate((test_y, test_X[:, (1-n_features):]), axis=1)
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,0]

  # calculate RMSE
  rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

  # calculate average error percentage
  avg = np.average(inv_y)
  error_percentage = rmse / avg

  if verbose:
    print("")
    print("Test Root Mean Square Error: %.3f" % rmse)
    print("Test Average Value for %s: %.3f" % (output_col_name, avg))
    print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))

  if draw_prediction_fit_plot:
    pyplot.plot(inv_y, label="Actual (%s)" % output_col_name)
    pyplot.plot(inv_yhat, label="Predicted (%s)" % output_col_name)
    pyplot.legend()
    pyplot.show()

  return (inv_y, inv_yhat, rmse, error_percentage)

# run the prediction
def run(dataset_file_path, dataset_header_row_index, dataset_index_col_name, dataset_col_to_predict, dataset_cols_to_drop, dataset_dropnan, scale_range, n_intervals, n_out_timesteps, train_percentage, n_neurons, n_batch, n_epochs, is_stateful, has_memory_stack, loss_function, optimizer_function, draw_loss_plot, draw_prediction_fit_plot, verbose):
  """
    dataset_file_path: the csv file path
    dataset_header_row_index: the header row index in the csv file
    dataset_index_col_name: the index column (can be None if no index is there)
    dataset_col_to_predict: the column name/index to predict
    dataset_cols_to_drop: the column names/indices to drop (single label or list-like)
    dataset_dropnan: whether to drop rows with NaN values in dataset after conversion to supervised learning
    scale_range: scale range to fit data in
    n_intervals: number of time lags (intervals) to use in each neuron
    n_out_timesteps: number of time-steps in future to predict
    train_percentage: percentage of train data related to the dataset series size; (1-train_percentage) will be for test data
    n_neurons: number of neurons for LSTM nn
    n_batch: nn batch size
    n_epochs: training epochs
    is_stateful: whether the model has memory states
    has_memory_stack: whether the model has memory stack
    loss_function: the model loss function evaluator
    optimizer_function: the loss optimizer function
    draw_loss_plot: whether to draw the loss history plot
    draw_prediction_fit_plot: whether to draw the the predicted vs actual fit plot
    verbose: whether to output some debug data
  """

  col_names, values, n_features, output_col_name = _load_dataset(dataset_file_path, dataset_header_row_index, dataset_index_col_name, dataset_col_to_predict, dataset_cols_to_drop)
  scaler, scaled = _scale_dataset(values, scale_range)
  supervised = _series_to_supervised(scaled, n_intervals, n_out_timesteps, dataset_dropnan, col_names, verbose)
  train_X, train_y, test_X, test_y = _split_data_to_train_test_sets(supervised.values, n_intervals, n_features, train_percentage, verbose)
  model, compatible_n_batch = _create_model(train_X, train_y, test_X, test_y, n_neurons, n_batch, n_epochs, is_stateful, has_memory_stack, loss_function, optimizer_function, draw_loss_plot, output_col_name, verbose)
  actual_target, predicted_target, error_value, error_percentage = _make_prediction(model, train_X, train_y, test_X, test_y, compatible_n_batch, n_intervals, n_features, scaler, draw_prediction_fit_plot, output_col_name, verbose)
  return (actual_target, predicted_target, error_value, error_percentage)


# if run directy from comman line
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Predict data from dataset")
  parser.add_argument("--dataset-file-path", default="dataset.csv", help="the csv file path")
  parser.add_argument("--n-intervals", default=3, help="number of time lags (intervals) to use in each neuron", type=int)
  parser.add_argument("--train-percentage", default=.30, help="percentage of train data related to the dataset series size", type=float)
  parser.add_argument("--n-neurons", default=50, help="number of neurons for LSTM nn", type=int)
  parser.add_argument("--n-batch", default=72, help="nn batch size", type=int)
  parser.add_argument("--n-epochs", default=60, help="training epochs", type=int)
  parser.add_argument("--is-stateful", default=False, help="whether the model has memory states", action="store_true")
  parser.add_argument("--has-memory-stack", default=False, help="whether the model has memory stack", action="store_true")
  parser.add_argument("--loss-function", default="mae", help="the model loss function evaluator")
  parser.add_argument("--optimizer-function", default="adam", help="the loss optimizer function")

  args = parser.parse_args()
  print("\nThe current setting used:\n=========================")
  print(json.dumps(vars(args), indent=4))

  start_time = time.clock()
  run(
    dataset_file_path = args.dataset_file_path,
    n_intervals = args.n_intervals,
    train_percentage = args.train_percentage,
    n_neurons = args.n_neurons,
    n_batch = args.n_batch,
    n_epochs = args.n_epochs,
    is_stateful = args.is_stateful,
    has_memory_stack = args.has_memory_stack,
    loss_function = args.loss_function,
    optimizer_function = args.optimizer_function,
    n_out_timesteps = 1,
    dataset_header_row_index = 0,
    dataset_index_col_name = "Id",
    dataset_col_to_predict = "CP",
    dataset_cols_to_drop = "IIN",
    dataset_dropnan = True,
    scale_range = (0, 1),
    draw_loss_plot = True,
    draw_prediction_fit_plot = True,
    verbose = True
  )
  print("\nExecuted in total of %f seconds" % (time.clock() - start_time))