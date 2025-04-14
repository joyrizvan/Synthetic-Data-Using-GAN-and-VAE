import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocess:
    def __init__(self):
        """
        Initialize the Preprocess class.
        Define the class attributs
        """
        self.original_file_path = "./data/original/"
        self.synthetic_file_path = "./data/synthetic/"
        self.scaler = StandardScaler()

    def run_preprocess(self, filename_real: str,filename_synthetic: str) -> pd.DataFrame:
        """
        Run the preprocessing steps on the data.
        Need to read synthetic data first and pass that when reading original data.
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        real_data = self.read_data(filename_real, False)
        synthetic_data = self.read_data(filename_synthetic, True)
        real_data = self.remove_null(real_data)
        synthetic_data = self.remove_null(synthetic_data)

        synthetic_data = self.remove_underscore(synthetic_data)
        real_data = self.restrict_columns(real_data, synthetic_data)
        return synthetic_data, real_data

    def remove_null(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        return df.dropna()

    def read_data(self, filename: str, is_synthetic: bool) -> pd.DataFrame:
        """
        Read data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: DataFrame containing the data from the CSV file.
        """
        try:
            if is_synthetic:
                file_path = self.synthetic_file_path + filename
            else:
                file_path = self.original_file_path + filename
            df = pd.read_csv(file_path)
            print("data loaded, shape - {}".format(df.shape))
            return df 
        except FileNotFoundError:
            print(f"File {filename} not found in {self.synthetic_file_path if is_synthetic else self.original_file_path}.")
            
    def remove_underscore(self,df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.replace("_", " ") for col in df.columns]

        return df

    def restrict_columns(self, real_training_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        real_training_data = real_training_data[synthetic_data.columns]
        return real_training_data

    def transform_data(self, df, target_column=None, categorical_features=None):
        df = df.copy()

        # Separate target if present
        target = df.pop(target_column) if target_column in df.columns else None

        # Auto-detect categorical features
        if categorical_features is None:
            categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Identify numerical features (exclude target)
        self.numerical_columns = df.drop(columns=categorical_features).select_dtypes(include=['number']).columns.tolist()

        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        # Only scale numerical features (after one-hot)
        df[self.numerical_columns] = self.scaler.fit_transform(df[self.numerical_columns])
        print("Data transformed, shape - {}".format(df.shape))
        return df, target
    def duplicate_real_data(self,real_df):
        """
        Duplicate the real dataset by concatenating it with itself.
        
        Args:
            real_df (pd.DataFrame): The real dataset.
            
        Returns:
            pd.DataFrame: Duplicated dataset (2x the original size).
        """
        return pd.concat([real_df, real_df.copy()], ignore_index=True)
    
    def augment_with_synthetic(self, real_df, synthetic_df):
        """
        Concatenate real data with synthetic data for augmentation.
        
        Args:
            real_df (pd.DataFrame): The real dataset.
            synthetic_df (pd.DataFrame): The synthetic dataset.
            
        Returns:
            pd.DataFrame: Combined dataset (real + synthetic).
        """
        return pd.concat([real_df, synthetic_df], ignore_index=True)
    
if __name__ == "__main__":
    preprocess = Preprocess()
    df = preprocess.read_data("customer_churn_dataset-testing-master.csv", False)
    target_column = "Churn"
    categorical_features = ['Gender', 'Subscription Type','Contract Length']
    df, target = preprocess.transform_data(df, target_column,categorical_features)
