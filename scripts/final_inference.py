import numpy as np
import tensorflow as tf
import os
from utils.utils import display_predictions

# Load data and model
DATA_PATH = "/content/data/processed"
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
model = tf.keras.models.load_model("model_3.keras")

# Predict class probabilities and labels
pred_probs = model.predict(X_test)
pred_classes = np.argmax(pred_probs, axis=1)
class_labels = {0: "Minor Damage", 1: "Severe Damage"}

# Format predictions with confidence
formatted = []
for i, (cls_idx, prob) in enumerate(zip(pred_classes, pred_probs)):
    label = class_labels[cls_idx]
    confidence = prob[cls_idx] * 100
    formatted.append(f"Image {i + 1}: {label} ({confidence:.2f}%)")

# Save to file
output_file = "final_predictions.txt"
with open(output_file, "w") as f:
    for line in formatted:
        f.write(line + "\n")

print(f"✅ Predictions saved to {output_file}")

# Optionally display top 10 with images
sample_images = X_test[:10]
sample_captions = [f"{class_labels[c]} ({pred_probs[i][c] * 100:.2f}%)" for i, c in enumerate(pred_classes[:10])]
display_predictions(sample_images, sample_captions)

if __name__ == "__main__":
    print("✅ Final inference complete.")
