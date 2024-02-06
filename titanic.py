"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------

import argparse

from import_data import import_yaml_config, import_data
from build_features import (
    create_variable_title,
    fill_na_titanic,
    label_encoder_titanic,
    check_has_cabin,
    ticket_length,
)
from train_evaluate import random_forest_titanic


parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument("--n_trees", type=int, default=20, help="Nombre d'arbres")
args = parser.parse_args()


# PARAMETRES -------------------------------

config = import_yaml_config("config.yaml")

API_TOKEN = config.get("jeton_api")
LOCATION_TRAIN = config.get("train_path", "train.csv")
LOCATION_TEST = config.get("test_path", "test.csv")
TEST_FRACTION = config.get("test_fraction", 0.1)
N_TREES = args.n_trees


# FEATURE ENGINEERING --------------------------------

TrainingData = import_data(LOCATION_TRAIN)
TestData = import_data(LOCATION_TEST)

# Create a 'Title' variable
TrainingData = create_variable_title(TrainingData)
TestData = create_variable_title(TestData)


# IMPUTATION DES VARIABLES ================

TrainingData = fill_na_titanic(TrainingData)
TestData = fill_na_titanic(TestData)

TrainingData = label_encoder_titanic(TrainingData)
TestData = label_encoder_titanic(TestData)


# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData = check_has_cabin(TrainingData)
TestData = check_has_cabin(TestData)

TrainingData = ticket_length(TrainingData)
TestData = ticket_length(TestData)


# MODELISATION: RANDOM FOREST ----------------------------

model = random_forest_titanic(
    data=TrainingData, fraction_test=TEST_FRACTION, n_trees=N_TREES
)
