import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score



def load_data(file_path):
    df = pd.read_csv(file_path)
    return df




def clean_data(df):

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove first column (user id)
    df = df.drop(df.columns[0], axis=1)

    # Separate columns by datatype
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    # Fill missing numeric values with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def feature_selection(X, y, k=10):

    k = min(k, X.shape[1])

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=k
    )

    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]

    print("\nSelected Features:")
    for feature in selected_features:
        print(feature)

    return X_selected


def prepare_data(df):

    target_column = "great_customer_class"

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_selected = feature_selection(X, y)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_selected)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    return accuracy, f1



def run_models(X_train, X_test, y_train, y_test):

    models = {

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "SVM":
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            ),

        "Logistic Regression":
            LogisticRegression(
                max_iter=5000,
                random_state=42
            ),

        "Naive Bayes":
            GaussianNB(),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=5
            )
    }

    results = {}

    print("\nMODEL RESULTS")
    print("-" * 60)

    for name, model in models.items():

        accuracy, f1 = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results[name] = accuracy

        print(
            f"{name:<20} "
            f"Accuracy = {accuracy:.4f} "
            f"F1 = {f1:.4f}"
        )

    return models, results


def run_voting_ensemble(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):

    voting_model = VotingClassifier(

        estimators=[
            ("rf", models["Random Forest"]),
            ("svm", models["SVM"]),
            ("lr", models["Logistic Regression"])
        ],

        voting="soft"
    )

    accuracy, f1 = evaluate_model(
        voting_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nENSEMBLE RESULT")
    print("-" * 60)

    print(
        f"Voting Ensemble "
        f"Accuracy = {accuracy:.4f} "
        f"F1 = {f1:.4f}"
    )

    return accuracy


def main():

    file_path = "https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv"

    df = load_data(file_path)

    print("Dataset Shape:", df.shape)

    df = clean_data(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    models, results = run_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    ensemble_accuracy = run_voting_ensemble(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_single_model_accuracy = max(results.values())

    print("\nCOMPARISON")
    print("-" * 60)

    print(
        f"Best Individual Model Accuracy: "
        f"{best_single_model_accuracy:.4f}"
    )

    print(
        f"Voting Ensemble Accuracy: "
        f"{ensemble_accuracy:.4f}"
    )

    if ensemble_accuracy > best_single_model_accuracy:
        print(
            "\nEnsemble learning improved the model performance."
        )
    elif ensemble_accuracy == best_single_model_accuracy:
        print(
            "\nEnsemble learning produced the same accuracy."
        )
    else:
        print(
            "\nEnsemble learning did not improve accuracy."
        )

    print(
        "\nAccuracy and F1-score were used as evaluation metrics."
    )


if __name__ == "__main__":
    main()
