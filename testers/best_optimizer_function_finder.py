from best_config_finder import run_tests

# if run directly from comman line
if __name__ == "__main__":
  best_result = run_tests(
    dataset_file_path = "../dataset.csv",
    output_file_path = "test_optimizer_function/result_log.txt",
    report_path_format = "test_optimizer_function/result_round_%s.xlsx",
    n_intervals_list = [3],
    train_percentage_list = [.5],
    n_neurons_list = [50],
    n_batch_list = [72],
    n_epochs_list = [60],
    is_stateful_list = [False],
    has_memory_stack_list = [False],
    loss_function_list = ["mean_squared_error"],
    optimizer_function_list = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]
  )
