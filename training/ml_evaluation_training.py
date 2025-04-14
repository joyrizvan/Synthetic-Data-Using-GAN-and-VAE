import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.preprocess import Preprocess


class ChurnModel:
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data
        self.target_column = target_column
        self.model_directory = "./model/"

    def train_logistic_regression(self, model_name="logistic_regression.pkl"):
        """"
        Train a Logistic Regression model and save it to the specified path.
        Data Transformed within the function.
        Args:
            model_name (str): Name of the model file to save.
        """
        pp = Preprocess()
        train_X_transformed, target = pp.transform_data(self.data, self.target_column)
        model_path = os.path.join(self.model_directory, model_name)

        if os.path.exists(model_path):
            print(f"Model already exists at {model_path}. Skipping training.")
        else:
            print("Training Logistic Regression model...")
            model = LogisticRegression()
            model.fit(train_X_transformed, target)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Logistic Regression model saved at {model_path}")

    def train_random_forest(self, model_name="random_forest.pkl"):
        """
        Train a Random Forest model and save it to the specified path.
        Data Transformed within the function.
        Args:
            model_path (str): Name of the model file to save.
        """
        pp = Preprocess()
        train_X_transformed, target = pp.transform_data(self.data, self.target_column)
        model_path = os.path.join(self.model_directory, model_name)
        if os.path.exists(model_path):
            print(f"Model already exists at {model_path}. Skipping training.")
        else:
            print("Training Random Forest model...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(train_X_transformed, target)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"✅ Random Forest model saved at {model_path}")

def run_training_logistic_scenario(scenario: int):
    pp = Preprocess()
    real_data_path = "imbalanced_customer_churn_dataset-training-master.csv"
    synthetic_tgan_path = "synthetic_data_tgan.csv"
    synthetic_tvae_path = "synthetic_data_tvae.csv"
    target = "Churn"

    synthetic_tgan, real_data = pp.run_preprocess(real_data_path, synthetic_tgan_path)
    synthetic_tvae, _ = pp.run_preprocess(real_data_path, synthetic_tvae_path)

    if scenario == 1:
        # Real data only
        churn_model = ChurnModel(real_data, target)
        churn_model.train_logistic_regression(model_name="real_data_logistic_regression.pkl")

    elif scenario == 2:
        # Synthetic TGAN only
        churn_model = ChurnModel(synthetic_tgan, target)
        churn_model.train_logistic_regression(model_name="synthetic_tgan_logistic_regression.pkl")

    elif scenario == 3:
        # Duplicated real data
        duplicated = pp.duplicate_real_data(real_data)
        churn_model = ChurnModel(duplicated, target)
        churn_model.train_logistic_regression(model_name="duplicated_real_data_logistic_regression.pkl")

    elif scenario == 4:
        # Augmented real + TGAN synthetic
        augmented = pp.augment_with_synthetic(real_data, synthetic_tgan)
        churn_model = ChurnModel(augmented, target)
        churn_model.train_logistic_regression(model_name="augmented_tgan_logistic_regression.pkl")

    elif scenario == 5:
        # Synthetic TVAE only
        churn_model = ChurnModel(synthetic_tvae, target)
        churn_model.train_logistic_regression(model_name="synthetic_tvae_logistic_regression.pkl")

    elif scenario == 6:
        # Augmented real + TVAE synthetic
        augmented = pp.augment_with_synthetic(real_data, synthetic_tvae)
        churn_model = ChurnModel(augmented, target)
        churn_model.train_logistic_regression(model_name="augmented_tvae_logistic_regression.pkl")

    else:
        print("❌ Invalid scenario selected. Choose from 1 to 6.")

def run_training_forest_scenario(scenario: int):
    pp = Preprocess()
    real_data_path = "imbalanced_customer_churn_dataset-training-master.csv"
    synthetic_tgan_path = "synthetic_data_tgan.csv"
    synthetic_tvae_path = "synthetic_data_tvae.csv"
    target = "Churn"

    synthetic_tgan, real_data = pp.run_preprocess(real_data_path, synthetic_tgan_path)
    synthetic_tvae, _ = pp.run_preprocess(real_data_path, synthetic_tvae_path)

    if scenario == 1:
        # Real data only
        churn_model = ChurnModel(real_data, target)
        churn_model.train_random_forest(model_name="real_data_random_forest.pkl")

    elif scenario == 2:
        # Synthetic TGAN only
        churn_model = ChurnModel(synthetic_tgan, target)
        churn_model.train_random_forest(model_name="synthetic_tgan_random_forest.pkl")

    elif scenario == 3:
        # Duplicated real data
        duplicated = pp.duplicate_real_data(real_data)
        churn_model = ChurnModel(duplicated, target)
        churn_model.train_random_forest(model_name="duplicated_real_data_random_forest.pkl")

    elif scenario == 4:
        # Augmented real + TGAN synthetic
        augmented = pp.augment_with_synthetic(real_data, synthetic_tgan)
        churn_model = ChurnModel(augmented, target)
        churn_model.train_random_forest(model_name="augmented_tgan_random_forest.pkl")

    elif scenario == 5:
        # Synthetic TVAE only
        churn_model = ChurnModel(synthetic_tvae, target)
        churn_model.train_random_forest(model_name="synthetic_tvae_random_forest.pkl")

    elif scenario == 6:
        # Augmented real + TVAE synthetic
        augmented = pp.augment_with_synthetic(real_data, synthetic_tvae)
        churn_model = ChurnModel(augmented, target)
        churn_model.train_random_forest(model_name="augmented_tvae_random_forest.pkl")

    else:
        print("❌ Invalid scenario selected. Choose from 1 to 6.")
if __name__ == "__main__":
    scenerios = [1,2,3,4,5,6]
    # for scenario in scenerios:
    for scenario in scenerios:
        print(f"Scenario {scenario}:")
        run_training_logistic_scenario(scenario)
    #run_training_logistic_scenario(2)