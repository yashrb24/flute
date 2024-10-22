import pandas as pd
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import os
from parser import parse_file, process_file
from multiprocessing import Pool
from functools import partial
from models import random_forest_classifier, svm_classifier, knn_classifier, xgboost_classifier, mlp_classifier

label_map = {chr(i): i - 97 for i in range(97, 123)}
label_map["noise"] = 0


def split_files(files_dict: dict, leave_out: str) -> tuple:
    """
    Splits the files into train and test sets based on the leave_out prefix.

    :param files_dict: Dictionary of {filename: processed_data}.
    :param leave_out: Prefix of the files to be left out for testing.
    :return: Tuple containing (train_dict, test_dict) where each dict has file as the key and processed data as the value.
    """
    train_df = []
    train_df_labels = []
    test_df = []
    test_df_labels = []

    for file, (data, label) in files_dict.items():
        if file.startswith(leave_out):
            test_df.append(data)
            test_df_labels.append(label)
        else:
            train_df.append(data)
            train_df_labels.append(label)

    train_df = pd.concat(train_df)
    train_df_labels = pd.concat(train_df_labels).applymap(lambda x: label_map.get(x, x))

    test_df = pd.concat(test_df)
    test_df_labels = pd.concat(test_df_labels).applymap(lambda x: label_map.get(x, x))

    return train_df, train_df_labels, test_df, test_df_labels


def process_single_file(file: str, data_dir: str) -> tuple:
    """
    Process a single file by parsing and processing its data.

    :param file: File name to process.
    :param data_dir: Directory where the files are stored.
    :return: Tuple containing the file name and processed data.
    """
    file_path = os.path.join(data_dir, file)
    df = parse_file(file_path)
    return file, process_file(df)


def train_and_evaluate_models(train_df, train_labels, test_df, test_labels):
    """
    Train and evaluate the classifiers.

    :param train_df: Training data.
    :param train_labels: Labels for training data.
    :param test_df: Test data.
    :param test_labels: Labels for test data.
    :return: Dictionary of classification reports for each model.
    """
    reports = {}

    # Train and evaluate Random Forest
    report_rf = random_forest_classifier(train_df, train_labels, test_df, test_labels)
    reports['Random Forest'] = report_rf

    # Train and evaluate SVM
    report_svm = svm_classifier(train_df, train_labels, test_df, test_labels)
    reports['SVM'] = report_svm

    # Train and evaluate KNN
    report_knn = knn_classifier(train_df, train_labels, test_df, test_labels)
    reports['KNN'] = report_knn

    # Train and evaluate XGBoost
    report_xgb = xgboost_classifier(train_df, train_labels, test_df, test_labels)
    reports['XGBoost'] = report_xgb

    # Train and evaluate MLP
    report_mlp = mlp_classifier(train_df, train_labels, test_df, test_labels)
    reports['MLP'] = report_mlp

    return reports


def average_reports(all_reports):
    """
    Compute the average of the reports across multiple leave-out tests, summing 'support' and averaging other metrics.

    :param all_reports: List of dictionaries of classification reports.
    :return: DataFrame with the average report.
    """
    report_summaries = {}

    for report_dict in all_reports:
        for model, report in report_dict.items():
            report_df = pd.DataFrame(report).transpose()
            
            if model not in report_summaries:
                # Initialize the report summary with the first report
                report_summaries[model] = report_df
            else:
                # Sum 'support' and add the other metrics
                report_summaries[model]['support'] += report_df['support']
                report_summaries[model].loc[:, report_df.columns != 'support'] += report_df.loc[:, report_df.columns != 'support']

    # Compute the average for all metrics except 'support' by dividing by the number of reports
    num_reports = len(all_reports)
    for model in report_summaries:
        report_summaries[model].loc[:, report_summaries[model].columns != 'support'] /= num_reports

    return report_summaries



def run_experiment(data_dir, leave_out_values):
    """
    Run the experiment for different leave-out values.

    :param data_dir: Directory containing the data files.
    :param leave_out_values: List of prefixes to leave out for testing.
    :return: Average classification report across all leave-out tests.
    """
    # Read all files from the data directory
    files = os.listdir(data_dir)

    # Determine the number of CPU cores to use
    num_cores = os.cpu_count()

    # Create a partial function with fixed data_dir
    process_func = partial(process_single_file, data_dir=data_dir)

    # Process all files in parallel and store them in a combined dictionary
    print("Reading data.....")
    with Pool(processes=num_cores) as pool:
        all_results = dict(tqdm(pool.imap(process_func, files), total=len(files)))

    all_reports = []

    for leave_out in leave_out_values:
        print(f"Running experiment with leave_out: {leave_out}")
        train_df, train_labels, test_df, test_labels = split_files(all_results, leave_out)

        # Uncomment this to use SMOTE for resampling
        train_df, train_labels = SMOTE().fit_resample(train_df, train_labels)

        train_labels = train_labels[0].values
        test_labels = test_labels[0].values

        # Train and evaluate models
        reports = train_and_evaluate_models(train_df, train_labels, test_df, test_labels)
        all_reports.append(reports)

    # Compute the average report
    avg_reports = average_reports(all_reports)

    # Print the average report for each model
    for model, report in avg_reports.items():
        print("=" * 100)
        print(f"Average Report for {model}:")
        print(report)

    return avg_reports


if __name__ == '__main__':
    data_dir = "data"
    leave_out_values = ["test_1", "test_2", "test_3", "test_4", "test_5"]

    # Run experiment and print average reports
    avg_reports = run_experiment(data_dir, leave_out_values)
