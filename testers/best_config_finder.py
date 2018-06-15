import sys
sys.path.insert(0,'..')

import time
import json
import os
from lstm import TimeSeriesPredictor

class Result:
  def __init__(self, setting, correct_direction_percentage, file_path):
    self.setting = setting
    self.correct_direction_percentage = correct_direction_percentage
    self.file_path = file_path

  def __str__(self):
    return "Setting:\n" + (json.dumps(self.setting, indent=4)) + "\nCorrect Direction Percentage: " + ("%.2f" % self.correct_direction_percentage) + "\nFile Path: " + self.file_path

def run_tests(dataset_file_path, output_file_path, report_path_format, n_intervals_list, train_percentage_list, n_neurons_list, n_batch_list, n_epochs_list, is_stateful_list, has_memory_stack_list, loss_function_list, optimizer_function_list):
  highest_correct_direction_percentage = -1
  best_setting = None
  best_result_file_path = None
  total_tests_count = len(n_intervals_list) * len(train_percentage_list) * len(n_neurons_list) * len(n_batch_list) * len(n_epochs_list) * len(is_stateful_list) * len(has_memory_stack_list) * len(loss_function_list) * len(optimizer_function_list)
  round = 0
  results = []

  for n_intervals in n_intervals_list:
    for train_percentage in train_percentage_list:
      for n_neurons in n_neurons_list:
        for n_batch in n_batch_list:
          for n_epochs in n_epochs_list:
            for is_stateful in is_stateful_list:
              for has_memory_stack in has_memory_stack_list:
                for loss_function in loss_function_list:
                  for optimizer_function in optimizer_function_list:
                    round += 1
                    setting = {
                      "n_intervals": n_intervals,
                      "train_percentage": train_percentage,
                      "n_neurons": n_neurons,
                      "n_batch": n_batch,
                      "n_epochs": n_epochs,
                      "is_stateful": is_stateful,
                      "has_memory_stack": has_memory_stack,
                      "loss_function": loss_function,
                      "optimizer_function": optimizer_function,
                    }
                    file_path = report_path_format % round
                    print("")
                    print("*** Round %d/%d ***" % (round, total_tests_count))
                    print(json.dumps(setting, indent=4))

                    start_time = time.clock()
                    tsp = TimeSeriesPredictor(
                      dataset_file_path = dataset_file_path,
                      n_intervals = n_intervals,
                      train_percentage = train_percentage,
                      n_neurons = n_neurons,
                      n_batch = n_batch,
                      n_epochs = n_epochs,
                      is_stateful = is_stateful,
                      has_memory_stack = has_memory_stack,
                      loss_function = loss_function,
                      optimizer_function = optimizer_function,
                      dropout = 0,
                      n_out_timesteps = 1,
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
                      output_excel_result = file_path,
                      verbose = False
                    )

                    actual_targets, predicted_targets, error_values, error_percentages, correct_direction_percentages, current_target_value, last_row_predicted_targets = tsp.run()
                    correct_direction_percentage = correct_direction_percentages[0] * 100
                    highest_correct_direction_percentage = max(highest_correct_direction_percentage, correct_direction_percentage)
                    if highest_correct_direction_percentage == correct_direction_percentage:
                      best_setting = setting
                      best_result_file_path = file_path
                    print("Correct Direction Percentage:", correct_direction_percentage)
                    print("Execution Duration:", time.clock() - start_time)
                    results.append(Result(setting, correct_direction_percentage, file_path))

  best_result = Result(best_setting, highest_correct_direction_percentage, best_result_file_path)
  print("\n\n*** Best Result ***\n")
  print(str(best_result))

  dirname = os.path.dirname(os.path.abspath(output_file_path))
  if not os.path.exists(dirname): os.makedirs(dirname)

  with open(output_file_path, "w") as f:
    for result in results:
      f.write(str(result) + "\n\n")
    f.write("*** Best Result ***\n")
    f.write(str(best_result))

  return best_result

# if run directly from comman line
if __name__ == "__main__":
  best_result = run_tests(
    dataset_file_path = "../dataset.csv",
    output_file_path = "test_config/result_log.txt",
    report_path_format = "test_config/result_round_%s.xlsx",
    n_intervals_list = [1, 3],
    train_percentage_list = [.3, .4, .5],
    n_neurons_list = [50, 70],
    n_batch_list = [48, 72],
    n_epochs_list = [50, 60],
    is_stateful_list = [False],
    has_memory_stack_list = [False],
    loss_function_list = ["mae", "mean_squared_error"],
    optimizer_function_list = ["adam", "sgd"]
  )
