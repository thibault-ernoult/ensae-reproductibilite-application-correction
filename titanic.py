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

# Extraction et ajout de la variable titre
TrainingData["Title"] = (
    TrainingData["Name"].apply(lambda x: x.split(",")[1]).apply(lambda x: x.split()[0])
)
TestData["Title"] = (
    TestData["Name"].apply(lambda x: x.split(",")[1]).apply(lambda x: x.split()[0])
)

# Suppression de la variable Titre
TrainingData.drop(labels="Name", axis=1, inplace=True)
TestData.drop(labels="Name", axis=1, inplace=True)

# Correction car Dona est présent dans le jeu de test à prédire mais
# pas dans les variables d'apprentissage
TestData["Title"] = TestData["Title"].replace("Dona.", "Mrs.")


fx, axes = plt.subplots(2, 1, figsize=(15, 10))
fig1_title = sns.countplot(data=TrainingData, x="Title", ax=axes[0]).set_title(
    "Fréquence des titres"
)
fig2_title = sns.barplot(
    data=TrainingData, x="Title", y="Survived", ax=axes[1]
).set_title("Taux de survie des titres")

# Age
sns.histplot(
    data=TrainingData, x='Age', bins=15, kde=False
).set_title("Distribution de l'âge")
plt.show()


## IMPUTATION DES VARIABLES ================


# Age
meanAge = round(TrainingData["Age"].mean())
TrainingData["Age"] = TrainingData["Age"].fillna(meanAge)
TestData["Age"] = TrainingData["Age"].fillna(meanAge)

# Sex, Title et Embarked
label_encoder_sex = LabelEncoder()
label_encoder_title = LabelEncoder()
label_encoder_embarked = LabelEncoder()
TrainingData["Sex"] = label_encoder_sex.fit_transform(TrainingData["Sex"].values)
TrainingData["Title"] = label_encoder_title.fit_transform(TrainingData["Sex"].values)
TrainingData["Embarked"] = label_encoder_embarked.fit_transform(
    TrainingData["Sex"].values
)


TrainingData["Embarked"] = TrainingData["Embarked"].fillna("S")
TestData["Embarked"] = TestData["Embarked"].fillna("S")
TestData["Fare"] = TestData["Fare"].fillna(TestData["Fare"].mean())

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
