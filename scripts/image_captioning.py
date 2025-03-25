import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from utils.utils import display_predictions

# Load processed data and model
DATA_PATH = "/content/data/processed"
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
model = tf.keras.models.load_model("model_3.keras")

# Make predictions
pred_probs = model.predict(X_test)
pred_classes = np.argmax(pred_probs, axis=1)

# Convert predictions to captions
class_labels = {0: "Minor Damage", 1: "Severe Damage"}
captions = [class_labels[c] for c in pred_classes]

# Display a sample of predictions
sample_images = X_test[:10]
sample_captions = captions[:10]
display_predictions(sample_images, sample_captions)

if __name__ == "__main__":
    print("✅ Image captioning complete.")
