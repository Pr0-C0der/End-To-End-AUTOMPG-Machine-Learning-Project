import requests

from utils import (
    load_data,
    remove_non_integer_values,
    add_parameters,
    one_hot_encoder,
    split_data,
    file_exists,
)

DATA_PATH = "./data/data.txt"
DATA_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)


def download_data(url=DATA_URL):
    """
    Description:
        This function downloads the data from a URL. 
        If the file already exists, it is overwritten.

    Args:
        url: The URL of the data to be downloaded.
    """

    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while requesting the data: {e}")

    if not file_exists(DATA_PATH):
        with open(DATA_PATH, "wb") as file:
            file.write(response.content)
        print("Data downloaded successfully.")
    else:
        print("Data already Present.")
        with open(DATA_PATH, "wb") as file:
            file.write(response.content)

        print("Rewriting the data.")


def data_preprocessing_pipeline():
    """
    Description:
        This function performs data preprocessing on the dataset. 
        The data is cleaned by removing non-integer values. 
        New parameters are added to the data. 
        The data is one-hot encoded. 
        The data is split into train and test sets.

    Returns:
        The train and test sets.
    """
    
    data = load_data(DATA_PATH)
    # Cleaning the data by removing non integer values
    processed_df = remove_non_integer_values(data)

    # Adding Parameters
    processed_df = add_parameters(processed_df)

    # One hot encodeing
    processed_df = one_hot_encoder(data)

    X_train, X_test, y_train, y_test = split_data(data)

    return X_train, X_test, y_train, y_test
