from best_config_finder import run_tests

# if run directly from comman line
if __name__ == "__main__":
  best_result = run_tests(
    dataset_file_path = "../dataset.csv",
    output_file_path = "test_loss_function/result_log.txt",
    report_path_format = "test_loss_function/result_round_%s.xlsx",
    n_intervals_list = [3],
    train_percentage_list = [.5],
    n_neurons_list = [50],
    n_batch_list = [72],
    n_epochs_list = [60],
    is_stateful_list = [False],
    has_memory_stack_list = [False],
    loss_function_list = ["mean_absolute_error", "mean_squared_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh", "kullback_leibler_divergence", "poisson", "cosine_proximity", "binary_crossentropy"],
    optimizer_function_list = ["adam"]
  )
