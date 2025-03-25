import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import os

def evaluate_model():
    # Load data
    DATA_PATH = "/content/data/processed"
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    # Load model
    model = tf.keras.models.load_model("model_3.keras")

    # Predict
    pred_probs = model.predict(X_test)
    pred_classes = np.argmax(pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    report = classification_report(y_true, pred_classes, digits=4)
    print("\nClassification Report:\n")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
