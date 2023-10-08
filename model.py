from sklearn.svm import SVR as SupportVectorRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

# Define parameter grids for each model
PARAM_GRIDS = {
    "SVR": {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
        "coef0": [0.01, 0.1, 10],
        "C": [0.001, 0.01, 0.1, 1, 10],
    },
    "RandomForestRegressor": {
        "n_estimators": [100, 200, 500, 1000],
        "max_features": ["sqrt", "log2"],
        "max_depth": [4, 6, 8, 10],
        "criterion": ["squared_error"],
    },
    "XGBRegressor": {
        "learning_rate": (0.05, 0.10, 0.15),
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2],
        "colsample_bytree": [0.3, 0.4],
    },
    "LGBMRegressor": {
        "reg_alpha": [0.1, 1, 10],
        "reg_lambda": [0.1, 1, 10],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 1],
        "n_estimators": [50, 100, 200],
    },
    "LinearRegression": {
        "copy_X": [True],
        "fit_intercept": [True, False],
        "positive": [True, False],
    },
}

SVR_MODEL = SupportVectorRegressor()
RF_MODEL = RandomForestRegressor(random_state=42)
XGB_MODEL = XGBRegressor(random_state=42)
LGBM_MODEL = LGBMRegressor(random_state=42)
LIR_MODEL = LinearRegression()


def svr_params():
    """
    Description:
        The SVR model is a support vector regression model. 
        The SVR parameter grid contains the parameters that will be used to tune the SVR model.

    Returns:
        The SVR model and the SVR parameter grid
    """
    return SVR_MODEL, PARAM_GRIDS["SVR"]


def rf_params():
    """
    Description:
        The RF model is a Random Forest Regressor model. 
        The RF parameter grid contains the parameters that will be used to tune the RF model.

    Returns:
        The RF model and the RF parameter grid.
    """
    return RF_MODEL, PARAM_GRIDS["RandomForestRegressor"]


def xgb_params():
    """
    Description:
        The XGB model is a XGBoost model. 
        The XGB parameter grid contains the parameters that will be used to tune the XGB model.

    Returns:
        The RF model and the RF parameter grid.
    """
    return XGB_MODEL, PARAM_GRIDS["XGBRegressor"]


def lgbm_params():
    """
    Description:
        The LGBM model is LightGBM model. 
        The LGBM parameter grid contains the parameters that will be used to tune the LGBM model.

    Returns:
        The LGBM model and the LGBM parameter grid.
    """
    return LGBM_MODEL, PARAM_GRIDS["LGBMRegressor"]


def lir_params():
    """
    Description:
        The LIR model is Linear Regression model. 
        The LIR parameter grid contains the parameters that will be used to tune the LIR model.

    Returns:
        The LIR model and the LIR parameter grid.
    """
    return LIR_MODEL, PARAM_GRIDS["LinearRegression"]
