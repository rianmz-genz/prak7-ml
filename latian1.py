import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_models(data_path):
    # Load dataset
    univ_data = pd.read_csv(data_path, delimiter=';')

    # Preprocessing
    univ_data['Target'] = univ_data['Target'].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})
    univ_data = univ_data.drop(columns=['Curricular units 1st sem (grade)', 'Tuition fees up to date',
                                        'Scholarship holder', 'Curricular units 2nd sem (enrolled)',
                                        'Curricular units 1st sem (enrolled)', 'Admission grade', 'Displaced',
                                        'Previous qualification (grade)', 'Curricular units 2nd sem (evaluations)',
                                        'Application order', 'Age at enrollment', 'Debtor', 'Gender', 'Nacionality',
                                        'Course', 'Curricular units 2nd sem (without evaluations)', 'GDP',
                                        'Application mode', 'Curricular units 1st sem (without evaluations)',
                                        'Curricular units 2nd sem (credited)', 'International',
                                        'Curricular units 1st sem (evaluations)', 'Inflation rate',
                                        'Educational special needs', 'Marital status', 'Previous qualification',
                                        'Mother\'s qualification', 'Mother\'s occupation', 'Father\'s occupation',
                                        'Father\'s qualification', 'Curricular units 1st sem (credited)',
                                        'Unemployment rate', 'Daytime/evening attendance\t'], axis=1)

    # Split data into features and target
    X = univ_data.drop("Target", axis=1)
    y = univ_data['Target']

    # Split data into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naïve Bayes": GaussianNB()
    }

    # Evaluate each model
    results = {}
    for model_name, model in models.items():
        model_results = {}

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        model_results["Accuracy"] = accuracy_score(y_test, y_pred)
        model_results["Precision"] = precision_score(y_test, y_pred, average='weighted')
        model_results["Recall"] = recall_score(y_test, y_pred, average='weighted')
        model_results["F1 Score"] = f1_score(y_test, y_pred, average='weighted')
        model_results["Confusion Matrix"] = confusion_matrix(y_test, y_pred)

        results[model_name] = model_results

    return results


# Evaluate models
data_path = "dataset/university_data.csv"
model_results = evaluate_models(data_path)


# Print results
for model_name, metrics in model_results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        if metric_name == "Confusion Matrix":
            print(f"{metric_name}:\n{value}")
        else:
            print(f"{metric_name}: {value:.4f}")
            
def compare_best_model(model_results):
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in model_results.items():
        accuracy = metrics["Accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    print(f"\nThe best model based on accuracy is: {best_model} with accuracy {best_accuracy:.4f}")

# Compare best model
compare_best_model(model_results)

