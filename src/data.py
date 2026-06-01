import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(path: str) -> pd.DataFrame:
    """
    Load the data from the given path and return a pandas DataFrame.
    Apply the required preprocessing steps

    path: str
    """
    df = pd.read_csv(path)
    mask = pd.to_numeric(df["TotalCharges"], errors="coerce").isna()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

def build_preprocessor() -> ColumnTransformer:
    """
    Preprocess the data by encoding categorical variables and scaling numerical features.

    df: pd.DataFrame
    """
    categorical_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]

    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ]
    )

    return preprocessor

