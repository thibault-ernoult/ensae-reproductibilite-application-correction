"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------
import os
import argparse
import yaml

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ARGUMENTS OPTIONNELS ----------------------------------------

parser = argparse.ArgumentParser(description="Nombre d'arbres")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres dans la random forest"
)
args = parser.parse_args()

# FONCTIONS ---------------------------------


def import_config_yaml(path: str) -> dict:
    """importer les paramètres d'un fichier yaml

    Args:
        path(str): le fichier de configuration .yaml

    Returns:
        dict: contient l'ensemble des paramètres définis par le fichier d'entrée
    """
    dict_config = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as stream:
            dict_config = yaml.safe_load(stream)
    return dict_config


def import_data(path: str) -> pd.DataFrame:
    """Import a .csv file and convert it into a DataFrame
    Drop the useless PassengerID column

    Args:
        path (str): chemin du fichier .csv

    Returns:
        pd.DataFrame: DataFrame utilisable
    """
    data = pd.read_csv(path)
    data = data.drop(columns="PassengerID")
    return data


def create_variable_title(df: pd.DataFrame) -> pd.DataFrame:
    """Create the "Title" variable and delete the "Name" one

    Args:
        df (pd.DataFrame): input DataFrame

    Returns:
        pd.DataFrame: DataFrame with Title and without Name
    """
    df["Title"] = df["Name"].str.split(",").str[1].str.split(".").str[0]
    df.drop(labels="Name", axis=1, inplace=True)
    # Correction car Dona est présent dans le jeu de test à prédire
    # ... mais n'est pas dans les variables d'apprentissage
    df["Title"] = df["Title"].replace("Dona.", "Mrs.")
    return df


def fillna_columns(
    df: pd.DataFrame, column: str = "Age", value: float = 0.0
) -> pd.DataFrame:
    """fill missing values in a column

    Args:
        df (pd.DataFrame): input DataFrame
        column (str, optional): column label. Defaults to "Age".
        value (float, optional): fill value. Defaults to 0.0.

    Returns:
        pd.DataFrame: filled DataFrame
    """
    df[column] = df[column].fillna(value)
    return df


def fillna_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline of imputations

    Args:
        df (pd.DataFrame): DataFrame to impute

    Returns:
        pd.DataFrame: imputed DataFrame
    """
    # Imputation de la variable Age
    meanAge = round(df["Age"].mean())
    df["Age"] = fillna_columns(df, "Age", meanAge)

    # Imputation de la variable Embarked
    df["Embarked"] = fillna_columns(df, "Embarked", "S")

    # Imputation de la variable Fare
    meanFare = df["Fare"].mean()
    df["Fare"] = fillna_columns(df, "Fare", meanFare)
    
    return df


# IMPORT DES PARAMETRES DU SCRIPT-------------------------------

config = import_config_yaml("config.yaml")

API_TOKEN = config.get("jeton_api")
TRAIN_PATH = config.get("train_path", "train.csv")
TEST_PATH = config.get("test_path", "test.csv")
TEST_FRACTION = config.get("test_fraction")

# IMPORT ET EXPLORATION DONNEES --------------------------------

TrainingData = import_data(TRAIN_PATH)
TestData = import_data(TEST_PATH)

# Classe
fig, axes = plt.subplots(
    1, 2, figsize=(12, 6)
)  # layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")

# Genre
print(
    TrainingData["Name"]
    .apply(lambda x: x.split(",")[1])
    .apply(lambda x: x.split()[0])
    .unique()
)


# FEATURE ENGINEERING --------------------------------


## VARIABLE 'Title' ===================

TrainingData = create_variable_title(TrainingData)
TestData = create_variable_title(TestData)


fx, axes = plt.subplots(2, 1, figsize=(15, 10))
fig1_title = sns.countplot(data=TrainingData, x="Title", ax=axes[0]).set_title(
    "Fréquence des titres"
)
fig2_title = sns.barplot(
    data=TrainingData, x="Title", y="Survived", ax=axes[1]
).set_title("Taux de survie des titres")

# Age
sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()


## IMPUTATION DES VARIABLES ================


# Age, Embarked and Fare
TrainingData = fillna_titanic(TrainingData)
TestData = fillna_titanic(TrainingData)

# Sex, Title
label_encoder_sex = LabelEncoder()
label_encoder_title = LabelEncoder()
label_encoder_embarked = LabelEncoder()
TrainingData["Sex"] = label_encoder_sex.fit_transform(TrainingData["Sex"].values)
TrainingData["Title"] = label_encoder_title.fit_transform(TrainingData["Sex"].values)
TrainingData["Embarked"] = label_encoder_embarked.fit_transform(
    TrainingData["Sex"].values
)


# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData["hasCabin"] = TrainingData.Cabin.notnull().astype(int)
TestData["hasCabin"] = TestData.Cabin.notnull().astype(int)
TrainingData.drop(labels="Cabin", axis=1, inplace=True)
TestData.drop(labels="Cabin", axis=1, inplace=True)


TrainingData["Ticket_Len"] = TrainingData["Ticket"].str.len()
TestData["Ticket_Len"] = TestData["Ticket"].str.len()
TrainingData.drop(labels="Ticket", axis=1, inplace=True)
TestData.drop(labels="Ticket", axis=1, inplace=True)


## SPLIT TRAIN/TEST ==================

y = TrainingData.iloc[:, 0].values
X = TrainingData.iloc[:, 1:12].values

# Feature Scaling
scaler_x = MinMaxScaler((-1, 1))
X = scaler_x.fit_transform(X)


# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION)


# MODELISATION: RANDOM FOREST ----------------------------


# Ici demandons d'avoir 20 arbres
rdmf = RandomForestClassifier(n_estimators=args.n_trees)
rdmf.fit(X_train, y_train)


# calculons le score sur le dataset d'apprentissage et sur le dataset de test
# (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = rdmf.score(X_test, y_test)
rdmf_score_tr = rdmf.score(X_train, y_train)
print(
    f"{round(rdmf_score * 100)} % de bonnes réponses sur les données de test pour validation \
        (résultat qu'on attendrait si on soumettait notre prédiction \
            sur le dataset de test.csv)"
)

print("matrice de confusion")
confusion_matrix(y_test, rdmf.predict(X_test))
