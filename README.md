# AUTOMPG Project

This project demonstrates an end-to-end application that predicts the mileage of a car based on various parameters. It includes the training and deployment of the machine learning models using the AutoMPG dataset.


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

[![Linkedin Badge](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/prathamesh-gadekar-b7352b245/)](https://www.linkedin.com/in/prathamesh-gadekar-b7352b245/)
[![Hotmail Badge](https://img.shields.io/badge/-Hotmail-0078D4?style=flat-square&logo=microsoft-outlook&logoColor=white&link=mailto:prathamesh.gadekar@hotmail.com)](mailto:prathamesh.gadekar@hotmail.com)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Pr0-C0der/exoplanet-detection/blob/main/LICENSE)
[![Python application](https://github.com/Pr0-C0der/End-To-End-AUTOMPG-Machine-Learning-Project/actions/workflows/main.yml/badge.svg)](https://github.com/Pr0-C0der/End-To-End-AUTOMPG-Machine-Learning-Project/actions/workflows/main.yml)


## Data

The dataset used for this project is the AutoMPG dataset, which contains information about various car models and their attributes. The data is downloaded from a provided link and is stored in the `data` folder.

## Data Preprocessing

The downloaded data is cleaned and preprocessed before training the models. The preprocessing steps include handling missing values, encoding categorical variables, and scaling numeric features. The data preprocessing functions are implemented in the `data_setup.py` file.

## Models

Five different regression models are used to predict the car mileage. These models are trained and evaluated on the dataset. The models used are:

- SupportVectorRegressor
- RandomForestRegressor
- XGBRegressor
- LGBMRegressor
- LinearRegression

The models and their respective hyperparameter grids are defined in the `model.py` file. Grid search is used for hyperparameter tuning to optimize the models' performance. Finally, a `Stacking Regressor` model is used stacking the output of individual estimator and use a regressor to compute the final prediction.

## Training

The `train.py` file is used to train the machine learning models. It utilizes the `click` module to build a command-line interface for easy training. To train a specific model, use the following command:

``` shell
  python train.py --model_name <model_name>

```
The `model_name` parameter specifies the model to train. To see the available model names, use the command:

``` shell
  python train.py --help

```
## Utility Functions

The `utils.py` file contains various utility functions that assist other files in performing their functions. These functions include data loading, saving models, query processing, etc.

## Testing

To ensure the correctness of the code, two testing files are provided: `test_train.py` and `test_utils.py`. These files test the functionality of the `train.py` and `utils.py` files, respectively.

## Makefile

A `Makefile` is provided to simplify the execution of commands and code files using the command-line interface. The following commands are available:

- `make install`: Installs the required modules.
- `make test`: Runs the test files to verify the code.
- `make format`: Formats the code files.
- `make lint`: Lints the code.
- `make apprun`: Runs the Flask app.
- `make trainmodel`: Trains the default model.
- `make all`: Install modules, lint, format and test the code in succession.

## Folders

The project includes the following folders:

- `templates`: Contains HTML files for the frontend design of the application.
- `static`: Contains CSS files for styling the frontend.
- `models`: Stores the trained machine learning models in pickle format.
- `data`: Stores the downloaded data in .txt format.

## Getting Started

To set up and run the project, follow these steps:

Clone the repository:
```shell
git clone https://github.com/Pr0-C0der/End-To-End-AUTOMPG-Machine-Learning-Project.git
```
You can refer to the `MakeFile` section for detailed instructions on using the available commands to perform desired operations on the code.

## Deployment

The application is deployed on the Render platform. It utilizes Flask to create a web-based interface for predicting car mileage based on the trained models. It provides an user-friendly interface to predict the mileage of a car based on various parameters.

To access the deployed app, please visit: [Deployed Application](https://autompg-ml-project.onrender.com/)


## Conclusion

This project provides an end-to-end solution for predicting car mileage based on various parameters. It demonstrates data preprocessing, model training, and deployment of the machine learning models. The user-friendly web interface allows users to conveniently input car parameters and obtain accurate mileage predictions. Feel free to explore the code, make modifications, and further enhance the capabilities of the application.
