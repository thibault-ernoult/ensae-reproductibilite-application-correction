"""
Fonctions d'import des paramètres et des données
"""
import os
import yaml
import pandas as pd


def import_yaml_config(filename: str = "toto.yaml") -> dict:
    """Import config.yaml

    Args:
        filename (str, optional): .yaml config file. Defaults to "toto.yaml".

    Returns:
        dict: config parameters in a Python dictionnary
    """
    dict_config = {}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as stream:
            dict_config = yaml.safe_load(stream)
    return dict_config


def import_data(path: str) -> pd.DataFrame:
    """Import Titanic datasets
    Args:
        path (str): File location
    Returns:
        pd.DataFrame: Titanic dataset
    """

    data = pd.read_csv(path)
    data = data.drop(columns="PassengerId")

    return data
