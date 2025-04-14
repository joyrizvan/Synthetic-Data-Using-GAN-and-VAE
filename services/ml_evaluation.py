import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from services.preprocess import Preprocess
def evaluate_model(model_path, data, target_column):
    pp = Preprocess()
    X_test, y_test = pp.transform_data(data, target_column)
    # Load the trained model from disk
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict labels and probabilities
    y_pred = model.predict(X_test)
    
    # Some models like SVM may not support predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))