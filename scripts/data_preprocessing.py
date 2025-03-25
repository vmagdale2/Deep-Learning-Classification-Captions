import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set paths
RAW_DATA_PATH = "/content/data/raw/aircraft_damage_dataset_v1"
OUTPUT_PATH = "/content/data/processed"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Parameters
IMG_SIZE = (224, 224)

# Classes
class_map = {
    "minor_damage": 0,
    "severe_damage": 1
}

def load_and_preprocess_images():
    images, labels = [], []
    for class_name, label in class_map.items():
        class_folder = os.path.join(RAW_DATA_PATH, class_name)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            try:
                img = load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Failed to load image: {img_path} | Error: {e}")
    return np.array(images), to_categorical(np.array(labels), num_classes=2)

if __name__ == "__main__":
    X, y = load_and_preprocess_images()
    print("✅ Data loaded and normalized")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"🔹 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Save to disk
    np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)

    print("✅ Preprocessed data saved to disk.")
