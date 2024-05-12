import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_models(data_path):
    # Load dataset
    sms_data = pd.read_csv(data_path)

    # Rename columns
    sms_data = sms_data.rename(columns={"v1": "label", "v2": "sms"})

    # Map label to binary (0 for ham, 1 for spam)
    sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})

    # Split data into features and target
    X = sms_data['sms']
    y = sms_data['label']

    # Vectorize SMS data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    # Split data into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naïve Bayes": MultinomialNB()
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
        model_results["Precision"] = precision_score(y_test, y_pred)
        model_results["Recall"] = recall_score(y_test, y_pred)
        model_results["F1 Score"] = f1_score(y_test, y_pred)
        model_results["Confusion Matrix"] = confusion_matrix(y_test, y_pred)

        results[model_name] = model_results

    return results


# Path to the dataset
data_path = "dataset/spam.csv"

# Evaluate models
model_results = evaluate_models(data_path)

# Print results
for model_name, metrics in model_results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        if metric_name == "Confusion Matrix":
            print(f"{metric_name}:\n{value}")
        else:
            print(f"{metric_name}: {value:.4f}")

def compare_models(results, metric="Accuracy"):
    best_model = None
    best_score = 0
    
    for model_name, metrics in results.items():
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_model = model_name
    
    print(f"\nThe best model based on {metric} is: {best_model} with {metric} {best_score:.4f}")
# Compare models based on accuracy
compare_models(model_results, metric="Accuracy")
