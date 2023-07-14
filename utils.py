import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np

TEST_SIZE = 0.2


def file_exists(file_path: str):
    """
    Description:
        This function checks if a file exists.

    Args:
        file_path: The path to the file to be checked.

    Returns:
        True if the file exists, False otherwise.
    """
    return os.path.exists(file_path)


def load_data(file_path):
    """
    Description:
        This function loads the data from a file. The file is in the format of the Auto MPG dataset. 
        The columns in the file are:
        mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin

    Args:
        file_path: The path to the file to be loaded.

    Returns:
        The loaded data.
    """
    cols = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
    ]
    # Load data from a file
    # pylint: disable=W0718
    try:
        data = pd.read_csv(
            file_path, sep=" ", comment="\t", skipinitialspace=True, names=cols
        )
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print(e)
        print("Error processing the data. Check if it is downloaded.")

    return data


def check_non_integer_values(data: pd.DataFrame, col_name: str):
    """
    Description:
        This function checks if there are any non-integer values in a column.
        If there are any non-integer values, they are returned as a list.

    Args:
        data: The data to be processed.
        col_name: The name of the column to be processed.

    Returns:
        The list of non-integer values in the column.

    """

    non_integer_values = data.loc[
        ~data[col_name].astype(str).str.match(r"[0-9]"), col_name
    ]

    return non_integer_values


def change_non_integer_values(data: pd.DataFrame, col_name: str, non_integer_values):
    """
    Description:
        The non-integer values are first replaced with the median of the remaining values in the column. Then, the column is converted to a float datatype.

    Args:
        data: The data to be processed.
        col_name: The name of the column to be processed.
        non_integer_values: The list of non-integer values in the column.

    Returns:
        The processed data.
    """
    # Imputing the non integer values in column with the median of remaining values
    numeric_values = data.loc[data[col_name].astype(str).str.match(r"[0-9]"), col_name]
    med = numeric_values.median()

    for i in range(len(non_integer_values)):
        data[col_name] = data[col_name].replace(non_integer_values.iloc[i], f"{med}")

    # Converting string variables of column to float datatype
    data[col_name] = data[col_name].astype(float)

    return data


def remove_non_integer_values(data: pd.DataFrame):
    """
    Description:
        This function removes non-integer values from the data. 
        This is done to ensure that the data is in a format that can be used by the machine learning models.

    Args:
        data: The data to be processed.

    Returns:
        The processed data.
    """

    columns = list(data.columns)

    for column in columns:
        non_integer_values = check_non_integer_values(data, column)

        if len(non_integer_values) > 0:
            data = change_non_integer_values(data, column, non_integer_values)

    return data


def add_parameters(data):
    """
    Description:
        This function adds four new parameters to the data: power_to_weight, acceleration_efficiency, displacement_per_cylinder, and weight_per_cylinder. 
        These parameters are calculated using the existing data in the data frame.

    Args:
        data: The data to be processed.

    Returns:
        The processed data.
    """

    # pylint: disable=W0718
    try:
        power_to_weight = data["horsepower"] / data["weight"]
        data["power_to_weight"] = power_to_weight

        acceleration_efficiency = data["acceleration"] / data["horsepower"]
        data["acceleration_efficiency"] = acceleration_efficiency

        displacement_per_cylinder = data["displacement"] / data["cylinders"]
        data["displacement_per_cylinder"] = displacement_per_cylinder

        weight_per_cylinder = data["weight"] / data["cylinders"]
        data["weight_per_cylinder"] = weight_per_cylinder

    except KeyError as e:
        print(f"KeyError: {e} not found in the dataframe.")
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return data


def standard_scaler(data: pd.DataFrame):
    """
    Description:
        Scale the data using StandardScaler.

    Args:
        data: The data to be scaled.

    Returns:
        The scaled data.
    """

    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(data)

    return X


def split_data(data: pd.DataFrame):
    """
    Description:
        This function splits the data into train and test sets using the `train_test_split` function. The data is also scaled using the `standard_scaler` function.

    Args:
        data: The data to be split.

    Returns:
        The train and test sets.
    """

    data = data.sample(frac=1)

    # Separate the labels
    y = data["mpg"]
    X = data.drop("mpg", axis=1)

    X = standard_scaler(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    return X_train, X_test, y_train, y_test


def save_model(model_name: str, model):
    """
    Description:
        This function saves a model as a pickle file.

    Args:
        model_name: The name of the model file.
        model: The model to be saved.
    """
    print("-------------------------SAVING STATUS-------------------------")
    modelPath = f"./models/{model_name}.pkl"

    if not file_exists(modelPath):
        with open(modelPath, "wb") as f:
            pickle.dump(model[0], f)

        print("Model saved successfully.\n\n")

    else:
        print("Model already Present.")

        with open(modelPath, "wb") as f:
            pickle.dump(model[0], f)

        print("Model retrained and saved successfully.\n\n")


def load_model(folder_path, model_filename):
    """
    Description:
        This function loads a model from a file. The model file is stored as a pickle file.

    Args:
        folder_path: The folder path where the model file is located.
        model_filename: The name of the model file.

    Returns:
        The loaded model
    """

    model_path = os.path.join(folder_path, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_filename}' not found in folder '{folder_path}'"
        )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


def convert_to_dict(keys, values):
    """
    Convert two lists into a key-value pair dictionary.

    Args:
        keys: The list of keys.
        values: The list of values.

    Returns:
        A dictionary with the keys and values.
    """

    # pylint: disable=W0622
    dict = {}
    for i in range(len(keys)):
        dict[keys[i]] = values[i]
    return dict


def convert_dictionary_values(query):
    """
    Convert dictionary values to int64, float64, float64, float64, float64, int64, int64 respectively.

    Args:
        dictionary: The dictionary to convert.

    Returns:
        The converted dictionary.
    """

    converted_dictionary = {}
    for key, value in query.items():
        if isinstance(value, str):
            try:
                converted_value = int(value, 64)
            except ValueError:
                converted_value = float(value)
        elif isinstance(value, float):
            converted_value = float(value)
        else:
            converted_value = value
        converted_dictionary[key] = converted_value
    return converted_dictionary


def one_hot_encoder(data: pd.DataFrame):
    """
    Description:
        This function one-hot encodes the origin feature by creating three new features, origin_1, origin_2, and origin_3. The value of each of these features is 1 if the origin of the car is the corresponding value, and 0 otherwise.

    Args:
        data: The data to be one-hot encoded.

    Returns:
        The one-hot encoded data.
    """

    origin_encoding = ["origin_1", "origin_2", "origin_3"]
    origin_value = int(data["origin"][0])

    for origin_enc in origin_encoding:
        if f"origin_{origin_value}" == origin_enc:
            data[origin_enc] = 1.0
        else:
            data[origin_enc] = 0.0

    return data


def query_processing(query):
    """
    Description:
        This function preprocess a query for prediction by converting dictionary values of query to suitable datatypes, adding parameters, one-hot encoding the query, and scaling the query.

    Args:
        query: The query to be preprocessed.

    Returns:
        The preprocessed query.
    """

    query = convert_dictionary_values(query)

    query = add_parameters(query)

    query = pd.DataFrame.from_dict([query])

    query = one_hot_encoder(query)

    query = query.iloc[0].values

    scaler = StandardScaler()
    query = scaler.fit_transform(query[:, np.newaxis])

    return query.T
