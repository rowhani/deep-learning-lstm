import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
import os
import argparse
import json
import string
from math import sqrt, ceil
from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat, ExcelWriter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional

class TimeSeriesPredictor:
  """
    General class untility for solving time series preciton problems
  """

  def __init__(self, dataset_file_path, dataset_header_row_index, dataset_index_col_name, dataset_col_to_predict, dataset_ignore_all_other_variables, dataset_cols_to_drop, dataset_dropnan, scale_range, n_intervals, n_out_timesteps, train_percentage, n_neurons, n_batch, n_epochs, is_stateful, has_memory_stack, dropout, loss_function, optimizer_function, draw_loss_plot, draw_prediction_fit_plot, draw_prediction_fit_plots_per_timestep, output_excel_result, verbose):
    """
      dataset_file_path: the csv file path
      dataset_header_row_index: the header row index in the csv file
      dataset_index_col_name: the index column (can be None if no index is there)
      dataset_col_to_predict: the column name/index to predict
      dataset_ignore_all_other_variables: drop all variables except the one to predict (assume as single variable prediction)
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
      dropout: model dropout
      loss_function: the model loss function evaluator
      optimizer_function: the loss optimizer function
      draw_loss_plot: whether to draw the loss history plot
      draw_prediction_fit_plot: whether to draw the the predicted vs actual fit plot
      draw_prediction_fit_plots_per_timestep: show distinct predicted vs actual fit plot for each future time step
      output_excel_result: file path to create csv file containing predicted values (can be None if you don't need report)
      verbose: whether to output some debug data
    """

    self.dataset_file_path = dataset_file_path
    self.dataset_header_row_index = dataset_header_row_index
    self.dataset_index_col_name = dataset_index_col_name
    self.dataset_col_to_predict = dataset_col_to_predict
    self.dataset_ignore_all_other_variables = dataset_ignore_all_other_variables
    self.dataset_cols_to_drop = dataset_cols_to_drop
    self.dataset_dropnan = dataset_dropnan
    self.scale_range = scale_range
    self.n_intervals = n_intervals
    self.n_out_timesteps = n_out_timesteps
    self.train_percentage = train_percentage
    self.n_neurons = n_neurons
    self.n_batch = n_batch
    self.n_epochs = n_epochs
    self.is_stateful = is_stateful
    self.has_memory_stack = has_memory_stack
    self.dropout = dropout
    self.loss_function = loss_function
    self.optimizer_function = optimizer_function
    self.draw_loss_plot = draw_loss_plot
    self.draw_prediction_fit_plot = draw_prediction_fit_plot
    self.draw_prediction_fit_plots_per_timestep = draw_prediction_fit_plots_per_timestep
    self.output_excel_result = output_excel_result
    self.verbose = verbose

    # intermediate/output variables
    self.n_features = None
    self.dataset_column_names = None
    self.dataset_original_values = None
    self.dataset_scaled_values = None
    self.dataset_supervised_values = None
    self.dataset_supervised_names = None
    self.output_col_name = None
    self.train_X = None
    self.train_y = None
    self.test_X = None
    self.test_y = None
    self.model = None
    self.losses = None
    self.val_losses = None
    self.compatible_n_batch = None
    self.scaled_predicted_target = None
    self.reshaped_test_X = None
    self.reshaped_test_y = None
    self.actual_targets = None
    self.predicted_targets = None
    self.current_actual_target = None
    self.scaled_last_row_for_prediction = None
    self.last_row_predicted_targets = None
    self.taget_averages = None
    self.error_values = None
    self.error_percentages = None
    self.correct_direction_percentages = None

  # load data set
  def _load_dataset(self):

    # read dataset from disk
    dataset = read_csv(self.dataset_file_path, header=self.dataset_header_row_index, index_col=False)

    # set index col
    if self.dataset_index_col_name:
      dataset.set_index(self.dataset_index_col_name, inplace=True)

    # drop all variables except the one to predict (assume as single variable prediction)
    if self.dataset_ignore_all_other_variables:
      if type(self.dataset_col_to_predict) == int:
        indices = filter(lambda x: x != self.dataset_col_to_predict, range(dataset.values.shape[0]))
        dataset.drop(index=indices, axis=0, inplace=True)
      else:
        col_names = filter(lambda x: x != self.dataset_col_to_predict, dataset.columns.values.tolist())
        dataset.drop(columns=col_names, axis=1, inplace=True)
    # drop nonused colums
    elif self.dataset_cols_to_drop:
      if type(self.dataset_cols_to_drop[0]) == int:
        dataset.drop(index=self.dataset_cols_to_drop, axis=0, inplace=True)
      else:
        dataset.drop(columns=self.dataset_cols_to_drop, axis=1, inplace=True)

    # get rows and column names
    col_names = dataset.columns.values.tolist()
    values = dataset.values

    # move the column to predict to be the first col:
    col_to_predict_index = self.dataset_col_to_predict if type(self.dataset_col_to_predict) == int else col_names.index(self.dataset_col_to_predict)
    output_col_name = col_names[col_to_predict_index]
    if col_to_predict_index > 0:
      col_names = [col_names[col_to_predict_index]] + col_names[:col_to_predict_index] + col_names[col_to_predict_index+1:]
      values = np.concatenate((values[:, col_to_predict_index].reshape((values.shape[0], 1)), values[:,:col_to_predict_index], values[:,col_to_predict_index+1:]), axis=1)

    # prev = DataFrame(values[:,0]).shift(1).values
    # slope = np.sign(values[:,0].reshape((values.shape[0], 1)) - prev)
    # values = np.concatenate((values, slope), axis=1)[1:,:]
    # col_names.append("%s_slope" % output_col_name)

    # ensure all data is float
    values = values.astype("float32")

    if self.verbose:
      print("\noriginal dataset values shape:", values.shape)
      print("features count:", values.shape[1])

    # save intermediate data
    self.n_features = values.shape[1]
    self.dataset_column_names = list(col_names)
    self.dataset_original_values = np.copy(values)
    self.output_col_name = output_col_name

    return (col_names, values, values.shape[1], output_col_name)

  # scale dataset
  def _scale_dataset(self, values):
    """
      values: dataset original values
    """

    # normalize features
    scaler = MinMaxScaler(feature_range=self.scale_range or (0, 1))
    scaled = scaler.fit_transform(values)

    # save intermediate data
    self.dataset_scaled_values = np.copy(scaled)

    return (scaler, scaled)

  # convert series to supervised learning (ex: var1(t)_row1 = var1(t-1)_row2)
  def _series_to_supervised(self, values, col_names):
    """
      values: dataset scaled values
      col_names: name of columns for dataset
    """

    n_vars = 1 if type(values) is list else values.shape[1]
    if col_names is None: col_names = ["var%d" % (j+1) for j in range(n_vars)]
    df = DataFrame(values)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(self.n_intervals, 0, -1):
      cols.append(df.shift(i))
      names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, self.n_out_timesteps):
      cols.append(df.shift(-i))
      if i == 0:
        names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
      else:
        names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if self.dataset_dropnan:
      agg.dropna(inplace=True)

    if self.verbose:
      print("\nsupervised dataset values shape:", agg.shape)
      print("*second dimension is calculated as: n_features(%d) x (n_intervals(%d) + n_out_timesteps(%d)) = %d" % (values.shape[1], self.n_intervals, self.n_out_timesteps, values.shape[1] * (self.n_intervals + self.n_out_timesteps)))

    # save intermediate data
    self.dataset_supervised_values = np.copy(agg.values)
    self.dataset_supervised_names = list(names)

    return agg

  # split into train and test sets
  def _split_data_to_train_test_sets(self, values, n_features):
    """
      values: dataset supervised values
      n_features: number of features (variables) per neuron
    """

    n_train_intervals = ceil(values.shape[0] * self.train_percentage)
    train = values[:n_train_intervals, :]
    test = values[n_train_intervals:, :]

    # split into input and outputs
    n_obs = self.n_intervals * n_features
    n_outs = [-n_features * j for j in range(self.n_out_timesteps, 0, -1)]
    train_X, train_y = train[:, :n_obs], train[:, n_outs[0]] if len(n_outs) == 1 else np.concatenate([train[:, j].reshape((train.shape[0], 1)) for j in n_outs], axis=1)
    test_X, test_y = test[:, :n_obs], test[:, n_outs[0]] if len(n_outs) == 1 else np.concatenate([test[:, j].reshape((test.shape[0], 1)) for j in n_outs], axis=1)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], self.n_intervals, n_features))
    test_X = test_X.reshape((test_X.shape[0], self.n_intervals, n_features))

    if self.verbose:
      print("")
      print("train_X shape:", train_X.shape)
      print("train_y shape:", train_y.shape)
      print("test_X shape:", test_X.shape)
      print("test_y shape:", test_y.shape)

    # save intermediate data
    self.train_X = np.copy(train_X)
    self.train_y = np.copy(train_y)
    self.test_X = np.copy(test_X)
    self.test_y = np.copy(test_y)

    return (train_X, train_y, test_X, test_y)

  # create the nn model
  def _create_model(self, train_X, train_y, test_X, test_y, output_col_name):
    """
      train_X: train inputs
      train_y: train targets
      test_X: test inputs
      test_y: test targets
      output_col_name: name of the output/target column to be predicted
    """

    # design network
    model = Sequential()

    # model.add(Dropout(self.dropout, input_shape=(train_X.shape[1], train_X.shape[2])))

    # save intermediate data
    self.compatible_n_batch = self.n_batch

    if self.is_stateful:
      # calculate new compatible batch size
      for i in range(self.compatible_n_batch, 0, -1):
        if train_X.shape[0] % i == 0 and test_X.shape[0] % i == 0:
          if self.verbose and i != self.compatible_n_batch:
            print ("\n*In stateful network, batch size should be dividable by training and test sets; had to decrease it to %d." % i)
          self.compatible_n_batch = i
          break

      model.add(LSTM(self.n_neurons, activation="relu", dropout=self.dropout, batch_input_shape=(self.compatible_n_batch, train_X.shape[1], train_X.shape[2]), stateful=True, return_sequences=self.has_memory_stack))
    else:
      model.add(LSTM(self.n_neurons, activation="relu", dropout=self.dropout, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=self.has_memory_stack))

    if self.has_memory_stack:
      model.add(LSTM(self.n_neurons, activation="relu", input_shape=(train_X.shape[1], train_X.shape[2]), stateful=self.is_stateful))

    model.add(Dense(self.n_out_timesteps))

    # from tensorflow.python.keras._impl.keras import backend as K
    # def slope_error(y_true, y_pred):
    #   # y_true_prev = DataFrame(y_true).shift(1).values
    #   # y_true_prev[0] = y_pred[0]
    #   y_true_prev = K.stack(y_true, axis=-1)
    #   return K.mean(K.abs(y_pred - y_true) - ((y_true - y_true_prev) * (y_pred - y_true_prev)), axis=-1)
    # self.loss_function = slope_error

    model.compile(loss=self.loss_function, optimizer=self.optimizer_function)

    if self.verbose:
      print("")

    # fit network
    losses = []
    val_losses = []
    if self.is_stateful:
      for i in range(self.n_epochs):
        history = model.fit(train_X, train_y, epochs=1, batch_size=self.compatible_n_batch, validation_data=(test_X, test_y), verbose=0, shuffle=False)

        if self.verbose:
          print("Epoch %d/%d" % (i + 1, self.n_epochs))
          print("loss: %f - val_loss: %f" % (history.history["loss"][0], history.history["val_loss"][0]))

        losses.append(history.history["loss"][0])
        val_losses.append(history.history["val_loss"][0])

        model.reset_states()
    else:
      history = model.fit(train_X, train_y, epochs=self.n_epochs, batch_size=self.compatible_n_batch, validation_data=(test_X, test_y), verbose=2 if self.verbose else 0, shuffle=False)

    if self.draw_loss_plot:
      pyplot.plot(history.history["loss"] if not self.is_stateful else losses, label="Train Loss (%s)" % output_col_name)
      pyplot.plot(history.history["val_loss"] if not self.is_stateful else val_losses, label="Test Loss (%s)" % output_col_name)
      pyplot.legend()
      pyplot.show()

    # save intermediate data
    self.model = model
    self.losses = losses if self.is_stateful else history.history["loss"]
    self.val_losses = val_losses if self.is_stateful else history.history["val_loss"]

    return (model, self.compatible_n_batch, self.losses, self.val_losses)

  # make a prediction
  def _make_prediction(self, model, train_X, train_y, test_X, test_y, compatible_n_batch, n_features):
    """
      model: nn model
      train_X: train inputs
      train_y: train targets
      test_X: test inputs
      test_y: test targets
      compatible_n_batch: modified (compatible) nn batch size
      n_features: number of features (variables) per neuron
    """

    if self.verbose:
      print("")

    # save intermediate data
    self.scaled_predicted_target = model.predict(test_X, batch_size=compatible_n_batch, verbose = 1 if self.verbose else 0)
    self.reshaped_test_X = test_X.reshape((test_X.shape[0], self.n_intervals*n_features))
    self.reshaped_test_y = test_y.reshape((len(test_y), self.n_out_timesteps))

    return (self.scaled_predicted_target, self.reshaped_test_X, self.reshaped_test_y)

  def _invert_scale(self, yhat, test_X, test_y, n_features, scaler):
    """
      yhat: scaled predicted target
      test_X: reshaped test inputs
      test_Y: reshaped test targets
      n_features: number of features (variables) per neuron
      scaler: the scaler object used to invert transformation to real scale
    """

    # save intermediate data
    self.actual_targets = None
    self.predicted_targets = None
    self.current_actual_target = None

    for i in range(self.n_out_timesteps):
      # invert scaling for forecast
      inv_yhat = np.concatenate((yhat[:,i].reshape((yhat.shape[0], 1)), test_X[:, (1-n_features):]), axis=1)
      inv_yhat = scaler.inverse_transform(inv_yhat)
      inv_yhat = inv_yhat[:,0].reshape((yhat.shape[0], 1))
      self.predicted_targets = inv_yhat if self.predicted_targets is None else np.concatenate((self.predicted_targets, inv_yhat), axis=1)

      # invert scaling for actual
      inv_y = np.concatenate((test_y[:,i].reshape((test_y.shape[0], 1)), test_X[:, (1-n_features):]), axis=1)
      inv_y = scaler.inverse_transform(inv_y)
      inv_y = inv_y[:,0].reshape((test_y.shape[0], 1))
      self.actual_targets = inv_y if self.actual_targets is None else np.concatenate((self.actual_targets, inv_y), axis=1)

    self.current_actual_target = scaler.inverse_transform(test_X[:, -n_features:])[:,0].reshape((test_X.shape[0], 1))

    return (self.actual_targets, self.predicted_targets, self.current_actual_target)

  def _calculate_errors(self, inv_y, inv_yhat, current_actual_target, output_col_name):
    """
      inv_y: inverted actual targets
      inv_yhat: inverted predicted targets
      current_actual_target: current values for the target (t0)
      output_col_name: name of the output/target column to be predicted
    """

    # save intermediate data
    self.taget_averages = []
    self.error_values = []
    self.error_percentages = []
    self.correct_direction_percentages = []

    for i in range(self.n_out_timesteps):
      # calculate RMSE
      rmse = sqrt(mean_squared_error(inv_y[:,i], inv_yhat[:,i]))

      # calculate average error percentage
      avg = np.average(inv_y[:,i])
      error_percentage = abs(rmse / avg)

      # calculate correct direction predictions
      count = inv_y.shape[0]
      diff_base = inv_y[:,i-1] if i > 0 else current_actual_target[:,0]
      actual_diff = inv_y[:,i] - diff_base
      prediction_diff = inv_yhat[:,i] - diff_base
      directions = ["TRUE" if np.sign(actual_diff[j]) == np.sign(prediction_diff[j]) else "FALSE" for j in range(count)]
      correct_direction_percentage = len(list(filter(lambda x: x == "TRUE", directions))) / count

      self.error_values.append(rmse)
      self.taget_averages.append(avg)
      self.error_percentages.append(error_percentage)
      self.correct_direction_percentages.append(correct_direction_percentage)

      if self.draw_prediction_fit_plot:
        pyplot.plot(inv_y[:,i], label="Actual %s (t%d)" % (output_col_name, i+1))
        pyplot.plot(inv_yhat[:,i], label="Predicted %s (t%d)" % (output_col_name, i+1))
        if self.draw_prediction_fit_plots_per_timestep:
          pyplot.legend()
          pyplot.show()

    if self.verbose:
      print("")
      print("Test Root Mean Square Error(s):", ", ".join(["%.3f" % rmse for rmse in self.error_values]))
      print("Test Average Value(s):", ", ".join(["%.3f" % avg for avg in self.taget_averages]))
      print("Test Average Error Percentage(s):", ", ".join(["%.2f/100.00" % (err * 100) for err in self.error_percentages]))
      print("Test Correct Directions Percentage(s)", ", ".join(["%.2f/100.00" % (dir * 100) for dir in self.correct_direction_percentages]))

    if self.draw_prediction_fit_plot and not self.draw_prediction_fit_plots_per_timestep:
      pyplot.legend()
      pyplot.show()

    return (self.error_values, self.error_percentages, self.correct_direction_percentages)

  # get last row with required previous values to predict (t-n, ..., t)
  def _get_last_row_for_prediction(self, values, n_features):
    """
      values: dataset scaled values
      n_features: number of features (variables) per neuron
    """

    df = DataFrame(values)
    cols = list()

    # input sequence (t-n, ... t-1)
    for i in range(self.n_intervals-1, 0, -1):
      cols.append(df.shift(i))

    # current sequence (t)
    cols.append(df.shift(0))

    # put it all together
    agg = concat(cols, axis=1)
    agg.dropna(inplace=True)

    # save intermediate data
    self.scaled_last_row_for_prediction = agg.values[-1].reshape((1, self.n_intervals, n_features))

    return self.scaled_last_row_for_prediction

  # convert last row to non scaled value
  def _invert_last_row_scale(self, predicted_y, last_row_X, n_features, scaler):
    """
      predicted_y: scaled predicted target
      last_row_X: reshaped scaled last row inputs
      scaler: the scaler object used to invert transformation to real scale
    """

    # save intermediate data
    self.last_row_predicted_targets = None

    last_row_X_reshaped =  last_row_X[-1, -1, (1-n_features):].reshape((1, n_features-1)) if n_features > 1 else last_row_X[-1, -1, (1-n_features):].reshape((1,1))
    for i in range(self.n_out_timesteps):
      predicted_y_reshaped = predicted_y[:,i].reshape((predicted_y.shape[0], 1))
      inv_predicted_y = np.concatenate((predicted_y_reshaped, last_row_X_reshaped), axis=1)
      inv_predicted_y = scaler.inverse_transform(inv_predicted_y)
      inv_predicted_y = inv_predicted_y[:,0].reshape((predicted_y.shape[0], 1))
      self.last_row_predicted_targets = inv_predicted_y if self.last_row_predicted_targets is None else np.concatenate((self.last_row_predicted_targets, inv_predicted_y), axis=1)

    return self.last_row_predicted_targets.reshape((self.n_out_timesteps,))

  # learn how to predict
  def _learn(self):
    col_names, values, n_features, output_col_name = self._load_dataset()
    scaler, scaled = self._scale_dataset(values)
    supervised = self._series_to_supervised(scaled, col_names)
    train_X, train_y, test_X, test_y = self._split_data_to_train_test_sets(supervised.values, n_features)
    model, compatible_n_batch, losses, val_losses = self._create_model(train_X, train_y, test_X, test_y, output_col_name)
    scaled_predicted_target, reshaped_test_X, reshaped_test_y = self._make_prediction(model, train_X, train_y, test_X, test_y, compatible_n_batch, n_features)
    actual_targets, predicted_targets, current_actual_target = self._invert_scale(scaled_predicted_target, reshaped_test_X, reshaped_test_y, n_features, scaler)
    error_values, error_percentages, correct_direction_percentages = self._calculate_errors(actual_targets, predicted_targets, current_actual_target, output_col_name)
    return (actual_targets, predicted_targets, current_actual_target, output_col_name, losses, val_losses, error_values, error_percentages, correct_direction_percentages, model, scaled, scaler, n_features, output_col_name)

  # last row prediction (after learn)
  def _predict(self, model, scaled, scaler, n_features, output_col_name):
    """
      model: nn model
      scaled: dataset scaled values
      scaler: the scaler object used to invert transformation to real scale
      n_features: number of features (variables) per neuron
      output_col_name: name of the output/target column to be predicted
    """
    scaled_last_row_for_prediction = self._get_last_row_for_prediction(scaled, n_features)
    scaled_predicted_y = model.predict(self.scaled_last_row_for_prediction, batch_size=1)
    last_row_predicted_targets = self._invert_last_row_scale(scaled_predicted_y, scaled_last_row_for_prediction, n_features, scaler)
    current_target_value = self.dataset_original_values[-1, 0]

    if self.verbose:
      print("\n\n*****************\n* Current Value *\n*****************")
      print("%s(t): %.3f" % (output_col_name, current_target_value))

      print("\n\n***************************\n* Next Predicted Value(s) *\n***************************")
      for i in range(self.n_out_timesteps):
        print("%s(t%d): %.3f" % (output_col_name, i+1, last_row_predicted_targets[i]))

    return (current_target_value, last_row_predicted_targets)

  # generate excel report
  def _generate_excel_report(self, current_target_value, last_row_predicted_targets, actual_targets, predicted_targets, current_actual_target, output_col_name, losses, val_losses):
    """
      current_target_value: current value of target (last row in dataset)
      last_row_predicted_targets: predicted values for the dataset (values that are predicted to come after the current value in last row)
      actual_targets: actual values for the target
      predicted_targets: predicted values for the target
      current_actual_target: current values for the target (t0)
      output_col_name: the target variable name
      losses: training losses
      val_losses: test losses
    """

    dirname = os.path.dirname(os.path.abspath(self.output_excel_result))
    if not os.path.exists(dirname): os.makedirs(dirname)

    writer = ExcelWriter(self.output_excel_result, engine="xlsxwriter")

    legend = {
      "A": ["Actual"],
      "P": ["Predicted"],
      "t": ["Time-step"],
      "Avg": ["Average"],
      "Err": ["Predicted to Actual Error"],
      "RMSE": ["Root Mean Square Error"],
      "Dir": ["Direction (Sign)"],
      "TRUE": ["Same Direction"],
      "FALSE": ["Opposite Direction"]
    }

    prediction = {}
    prediction["A(t)"] = [current_target_value]
    for i in range(self.n_out_timesteps):
      prediction["P(t%d)" % (i+1)] = [last_row_predicted_targets[i]]
    for i in range(self.n_out_timesteps):
      prediction["RMSE(t%d)" % (i+1)] = self.error_values[i]
    for i in range(self.n_out_timesteps):
      prediction["Err(t%d) %%" % (i+1)] = self.error_percentages[i] * 100

    learning_count = predicted_targets.shape[0]
    learning_output = {}
    learning_output["A(t0)"] = current_actual_target[:,0]
    learning_results = {}
    learning_results["Avg[A(t0)]"] = np.average(learning_output["A(t0)"])
    charts = {
      "%s - Loss" % output_col_name: {
        "count": self.n_epochs,
        "y_axis": "Loss",
        "x_axis": "Epoch",
        "dataset": {
          "Train": losses,
          "Test": val_losses
        }
      }
    }
    for i in range(self.n_out_timesteps):
      learning_output["SPACE%d" % (i+1)] = ["" for j in range(learning_count)]
      learning_output["A(t%d)" % (i+1)] = actual_targets[:,i]
      learning_output["P(t%d)" % (i+1)] = predicted_targets[:,i]
      learning_output["A(t%d)-A(t%d) %%" % (i+1, i)] = (learning_output["A(t%d)" % (i+1)] - learning_output["A(t%d)" % i]) * 100 / learning_output["A(t%d)" % i]
      learning_output["P(t%d)-A(t%d) %%" % (i+1, i)] = (learning_output["P(t%d)" % (i+1)] - learning_output["A(t%d)" % i]) * 100 / learning_output["A(t%d)" % i]
      learning_output["Err(t%d) %%" % (i+1)] = np.absolute(learning_output["P(t%d)-A(t%d) %%" % (i+1, i)] - learning_output["A(t%d)-A(t%d) %%" % (i+1, i)])
      learning_output["Dir(t%d)" % (i+1)] = ["TRUE" if np.sign(learning_output["A(t%d)-A(t%d) %%" % (i+1, i)][j]) == np.sign(learning_output["P(t%d)-A(t%d) %%" % (i+1, i)][j]) else "FALSE" for j in range(learning_count)]

      learning_results["SPACE%d" % (i+1)] = [""]
      learning_results["Avg[A(t%d)]" % (i+1)] = [np.average(learning_output["A(t%d)" % (i+1)])]
      learning_results["Avg[P(t%d)]" % (i+1)] = [np.average(learning_output["P(t%d)" % (i+1)])]
      learning_results["Avg[A(t%d)-A(t%d)] %%" % (i+1, i)] = [np.average(learning_output["A(t%d)-A(t%d) %%" % (i+1, i)])]
      learning_results["Avg[P(t%d)-A(t%d)] %%" % (i+1, i)] = [np.average(learning_output["P(t%d)-A(t%d) %%" % (i+1, i)])]
      learning_results["Avg[Err(t%d)] %%" % (i+1)] = [np.average(learning_output["Err(t%d) %%" % (i+1)])]
      learning_results["TRUE[t%d] %%" % (i+1)] = [len(list(filter(lambda x: x == "TRUE", learning_output["Dir(t%d)" % (i+1)]))) * 100 / learning_count]

      charts["%s(t%d) - Chart" % (output_col_name, i+1)] = {
        "count": learning_count,
        "y_axis": output_col_name,
        "x_axis": "Interval",
        "dataset": {
          "A(t%d)" % (i+1): actual_targets[:,i],
          "P(t%d)" % (i+1): predicted_targets[:,i]
        }
      }

    setting = {
      "n_intervals": [self.n_intervals],
      "train_percentage": [self.train_percentage],
      "n_neurons": [self.n_neurons],
      "n_batch": [self.n_batch],
      "n_epochs": [self.n_epochs],
      "is_stateful": [self.is_stateful],
      "has_memory_stack": [self.has_memory_stack],
      "loss_function": [self.loss_function],
      "optimizer_function": [self.optimizer_function]
    }

    learning_sheet_name = "%s - Learning" % output_col_name
    learning_results_items = map(lambda item: ("", item[1]) if item[0].startswith("SPACE") else item, learning_results.items())
    DataFrame.from_items(learning_results_items).to_excel(writer, learning_sheet_name, index=False, startrow=0)
    output_items = map(lambda item: ("", item[1]) if item[0].startswith("SPACE") else item, learning_output.items())
    DataFrame.from_items(output_items).to_excel(writer, learning_sheet_name, index=False, startrow=3)

    for sheet_name, data in charts.items():
      df = DataFrame(data["dataset"])
      df.to_excel(writer, sheet_name=sheet_name)
      workbook  = writer.book
      worksheet = writer.sheets[sheet_name]
      chart = workbook.add_chart({"type": "line"})
      lines_count = len(data["dataset"].keys())
      for i in range(lines_count):
        col = i + 1
        chart.add_series({
            "name": [sheet_name, 0, col],
            "categories": [sheet_name, 1, 0,   data["count"], 0],
            "values": [sheet_name, 1, col, data["count"], col],
        })
        chart.set_x_axis({"name": data["x_axis"]})
        chart.set_y_axis({"name": data["y_axis"], "major_gridlines": {"visible": False}})
        worksheet.insert_chart("%s2" % string.ascii_uppercase[lines_count + 2], chart)

    prediction_sheet_name = "%s - Prediction" % output_col_name
    DataFrame(prediction).to_excel(writer, prediction_sheet_name, index=False)

    legend_sheet_name = "Legend"
    DataFrame(legend).to_excel(writer, legend_sheet_name, index=False)

    setting_sheet_name = "Setting Parameters"
    DataFrame(setting).to_excel(writer, setting_sheet_name, index=False)

    learning_max_width = np.max(list(map(lambda key: len(key), list(learning_results.keys()) + list(learning_output.keys()))))
    writer.sheets[learning_sheet_name].set_column(0, len(learning_results.keys()) + len(learning_output.keys()), learning_max_width)
    writer.sheets[prediction_sheet_name].set_column(0, len(prediction.keys()), 13)
    writer.sheets[legend_sheet_name].set_column(0, len(legend.keys()), 24)
    writer.sheets[setting_sheet_name].set_column(0, len(setting.keys()), 20)

    writer.save()

  # run the prediction (learn, predict, and generate report)
  def run(self):
    actual_targets, predicted_targets, current_actual_target, output_col_name, losses, val_losses, error_values, error_percentages, correct_direction_percentages, model, scaled, scaler, n_features, output_col_name = self._learn()
    current_target_value, last_row_predicted_targets = self._predict(model, scaled, scaler, n_features, output_col_name)
    if self.output_excel_result: self._generate_excel_report(current_target_value, last_row_predicted_targets, actual_targets, predicted_targets, current_actual_target, output_col_name, losses, val_losses)
    return (actual_targets, predicted_targets, error_values, error_percentages, correct_direction_percentages, current_target_value, last_row_predicted_targets)


# if run directly from comman line
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
  parser.add_argument("--loss-function", default="mean_squared_error", help="the model loss function evaluator (can be mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge, logcosh, kullback_leibler_divergence, poisson, binary_crossentropy, or cosine_proximity)")
  parser.add_argument("--optimizer-function", default="adam", help="the loss optimizer function (can be ada, sgd, rmsprop, adagrad, adadelta, adamax, or nadam)")

  args = parser.parse_args()
  print("\nThe current setting used:\n=========================")
  print(json.dumps(vars(args), indent=4))

  start_time = time.clock()
  tsp = TimeSeriesPredictor(
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
    dropout = 0,
    n_out_timesteps = 4,
    dataset_header_row_index = 0,
    dataset_index_col_name = "Id",
    dataset_col_to_predict = "CP",
    dataset_ignore_all_other_variables = False,
    dataset_cols_to_drop = "IIN",
    dataset_dropnan = True,
    scale_range = (0, 1),
    draw_loss_plot = False,
    draw_prediction_fit_plot = False,
    draw_prediction_fit_plots_per_timestep = False,
    output_excel_result = "output/report.xlsx",
    verbose = True
  )
  tsp.run()
  print("\nExecuted in total of %f seconds" % (time.clock() - start_time))