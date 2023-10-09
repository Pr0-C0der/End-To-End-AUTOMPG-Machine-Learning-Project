import pytest
from train import (
    train_model,
    train_all_models,
)

from data_setup import data_preprocessing_pipeline


# Test train_model function
def test_train_model():
    """
    Tests the train_model function

    This function tests the train_model function by calling it with the "svr" model name and checking if the model is not None and the RMSE is greater than or equal to 0.0.

    Returns:
        None
    """

    X_train, X_test, y_train, y_test = data_preprocessing_pipeline()

    model_name = "svr"
    model, rmse = train_model(model_name, X_train, X_test, y_train, y_test)

    assert model is not None
    assert rmse >= 0.0

    # Add more assertions for other models


# Test train_all_models function
def test_train_all_models():
    """
    Tests the train_all_models function

    This function tests the train_all_models function by calling it and checking if the length of the list of models is greater than 0 and all of the models are not None.

    Returns:
        None
    """

    X_train, X_test, y_train, y_test = data_preprocessing_pipeline()

    models = train_all_models(X_train, X_test, y_train, y_test)

    assert len(models) > 0
    assert all(model is not None for model in models)


if __name__ == "__main__":
    pytest.main()
