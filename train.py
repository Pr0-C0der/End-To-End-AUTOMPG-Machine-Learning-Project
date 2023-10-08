from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import time
import click

from data_setup import data_preprocessing_pipeline
from utils import save_model
from model import lgbm_params, lir_params, svr_params, rf_params, xgb_params

# Dictionary to store model names, models, and parameter functions
MODELS = {
    "svr": (svr_params, "svr"),
    "lir": (lir_params, "lir"),
    "xgboost": (xgb_params, "xgboost"),
    "lightgbm": (lgbm_params, "lgbm"),
    "rf": (rf_params, "rf"),
}


def model_output(model, X_train, X_test, y_train, y_test):
    """
    Trains the model and evaluates performance of the given model. 

    Args:
        model: The model to be evaluated.
        X_train: The training data.
        X_test: The test data.
        y_train: The training labels.
        y_test: The test labels.

    Returns:
        The model and the RMSE.
    """
     
    print(f"For Model: {type(model).__name__}")
    # Step 1: Training the Model
    model.fit(X_train, y_train)

    # Step 2: Cross-Validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"
    )  # Perform 5-fold cross-validation

    # Step 3: Evaluate the Model
    y_pred = model.predict(X_test)

    # Print the cross-validation scores and test accuracy
    print("Cross-Validation Scores:", -cv_scores)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"RMSE : {rmse}")

    return model, rmse


def find_best_model(model, param_grid, X_train, X_test, y_train, y_test):
    """
    Searches for the best model using GridSearchCV.

    Args:
        model: The model to be evaluated.
        param_grid: The hyperparameters to be searched.
        X_train: The training data.
        X_test: The test data.
        y_train: The training labels.
        y_test: The test labels.

    Returns:
        The best model and the RMSE.
    """

    print(f"Training : {type(model).__name__}")

    CV_md = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
    )

    start = time.time()
    CV_md.fit(X_train, y_train)
    end = time.time()

    print("-------------------------REPORT-------------------------")
    print("Total Time Required : ", end - start)

    best = CV_md.best_estimator_  # Best Estimator
    print("Best Estimator:")
    print(str(best) + "\n")
    clf = best
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"RMSE : {rmse}\n")

    return best, rmse


def train_model(model_name, X_train, X_test, y_train, y_test):
    """
    Trains a model (tuning it using GridSearch) and saves it.
    To know which models are av

    Args:
        model_name: The name of the model to be trained.
        X_train: The training data.
        X_test: The test data.
        y_train: The training labels.
        y_test: The test labels.

    Returns:
        The model and the RMSE.
    """

    if model_name not in MODELS:
        print("Model not found!! Type --help to get model information.")
        return

    param_func, model_filename = MODELS[model_name]
    md, param_grid = param_func()

    model, rmse = find_best_model(
        model=md,
        param_grid=param_grid,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    save_model(model_filename, (model, rmse))

    return model, rmse


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Trains all models and returns the best models.

    Args:
        X_train: The training data.
        X_test: The test data.
        y_train: The training labels.
        y_test: The test labels.

    Returns:
        A list of the best models.

    Available models:
        - svr (Suport Vector)
        - lir (Linear Regression)
        - xgboost (XGBoost)
        - lightgbm (Light GBM)
        - rf (Random Forest)

    """

    ModelsandRMSE = {}

    # pylint: disable=W0612
    for model_name, (param_func, _) in MODELS.items():
        model, rmse = train_model(model_name, X_train, X_test, y_train, y_test)
        ModelsandRMSE[model] = rmse

    best_models = sorted(ModelsandRMSE.items(), key=lambda x: x[1])
    return [(f"{type(model).__name__}", model) for model, _ in best_models]


def train_stacking_classifier(X_train, X_test, y_train, y_test):
    estimators = train_all_models(X_train, X_test, y_train, y_test)

    model = StackingRegressor(estimators=estimators)

    st_model = model_output(model, X_train, X_test, y_train, y_test)

    save_model("st", st_model)


@click.command()
@click.option(
    "--model_name",
    default="st",
    help="Specify the model. The default is set to Stacking Regressor",
)
def train_model_cli(model_name="st"):
    """Program to help you train models.

    Args:
        --model:\n
        svr: "Support Vector Regressor"\n
        lir: "Linear Regression"\n
        xgboost: "XGB Regressor"\n
        lightgbm: "LGBM Regressor"\n
        rf: "Random Forest Regressor"\n
        st: "Stacking Classifier"\n

        NOTE: Stacking Classifier chooses 5 best performing models and stacks them.
    """
    X_train, X_test, y_train, y_test = data_preprocessing_pipeline()

    if model_name == "st":
        train_stacking_classifier(X_train, X_test, y_train, y_test)
    else:
        train_model(model_name, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_model_cli()
