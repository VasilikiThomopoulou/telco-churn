from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import joblib

from src.data import load_data, build_preprocessor

def main():
    #1. Load data
    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    #2. Seperate target and features
    X = df.drop(['Churn', 'customerID'], axis=1)
    y = (df['Churn'] == "Yes").astype(int)

    #3. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1997, stratify=y)

    #4. Build the pipeline
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("model", GradientBoostingClassifier(random_state=42, n_estimators=300, learning_rate=0.01, max_depth=7)),
    ])

    #5. Train the model
    pipe.fit(X_train, y_train)

    #6. Evaluate the model
    proba_train = pipe.predict_proba(X_train)[:, 1]
    proba_test = pipe.predict_proba(X_test)[:, 1]
    print("Train ROC_AUC:", round(roc_auc_score(y_train, proba_train), 4))
    print("Test ROC_AUC:", round(roc_auc_score(y_test, proba_test), 4))

    print("Train Accuracy:", round(accuracy_score(y_train, pipe.predict(X_train)), 4))
    print("Test Accuracy:", round(accuracy_score(y_test, pipe.predict(X_test)), 4))

    #7 Save the whole fitted pipeline as one artfact
    joblib.dump(pipe, "models/telco_churn_model.pkl")
    print("Model saved to model/telco_churn_model.pkl")


if __name__ == "__main__":
    main()