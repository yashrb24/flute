import pandas as pd
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import os
from parser import parse_file, process_file
from multiprocessing import Pool
from functools import partial
from models import random_forest_classifier, svm_classifier, knn_classifier, xgboost_classifier, mlp_classifier

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

    # create a dictionary to map each letter to its index starting from 1 in the alphabet, all lowercase
    label_map = {chr(i): i - 96 for i in range(97, 123)}
    label_map["noise"] = 0

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


if __name__ == '__main__':
    data_dir = "/home/candy/Projects/flute/data"

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

    leave_out = "test_5"
    train_df, train_labels, test_df, test_labels = split_files(all_results, leave_out)
    train_df_resampled, train_label_resampled = train_df, train_labels

    train_label_resampled = train_label_resampled[0].values
    test_labels = test_labels[0].values

    print("=" * 100)
    print("Class Frequency in Train Data:")
    # print(train_labels.value_counts())
    # print(f"Total number of samples in train data: {len(train_labels)}")

    # Resample the data using SMOTE
    train_df_rescaled, train_label_resampled = SMOTE().fit_resample(train_df, train_labels)
    print(train_label_resampled.value_counts())
    # print(f"Total number of samples in train data: {len(train_label_resampled)}")

    # report_rf = random_forest_classifier(train_df_resampled, train_label_resampled, test_df, test_labels)
    # print("=" * 100)
    # print("Random Forest Classifier Report:")
    # print((pd.DataFrame(report_rf).transpose()))

    # report_svm = svm_classifier(train_df_resampled, train_label_resampled, test_df, test_labels)
    # print("=" * 100)
    # print("SVM Classifier Report:")
    # print(pd.DataFrame(report_svm).transpose())

    # report_knn = knn_classifier(train_df_resampled, train_label_resampled, test_df, test_labels)
    # print("=" * 100)
    # print("KNN Classifier Report:")
    # print(pd.DataFrame(report_knn).transpose())

    # report_xgb = xgboost_classifier(train_df_resampled, train_label_resampled, test_df, test_labels)
    # print("=" * 100)
    # print("XGBoost Classifier Report:")
    # print(pd.DataFrame(report_xgb).transpose())

    report_mlp = mlp_classifier(train_df_resampled, train_label_resampled, test_df, test_labels)
    print("=" * 100)
    print("MLP Classifier Report:")
    print(pd.DataFrame(report_mlp).transpose())
