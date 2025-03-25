import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(data_path):
    """
    Load and preprocess data from the given path.
    """
    # Load data logic here
    pass

def preprocess_images(images):
    """
    Preprocesses images for model training.
    """
    images = images / 255.0  # Normalize to range [0, 1]
    return images

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and test sets for model evaluation.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_model(model, path):
    """
    Saves the trained model to the specified path.
    """
    model.save(path)


def load_model(path):
    """
    Loads the saved model from the specified path.
    """
    return tf.keras.models.load_model(path)


def display_predictions(images, captions):
    """
    Display a list of images with their corresponding predicted captions.
    Automatically handles grayscale or RGB and rescales only if needed.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    for idx, (img, caption) in enumerate(zip(images, captions)):
        display_img = img.copy()

        print(f"Image {idx} - Original shape: {display_img.shape}, min: {display_img.min()}, max: {display_img.max()}")

        # Handle grayscale images by converting to RGB
        if display_img.ndim == 2:
            display_img = np.stack([display_img] * 3, axis=-1)

        # Smart rescaling based on range
        if display_img.max() <= 0.01:
            display_img = (display_img * 255 * 255).astype("uint8")
        elif display_img.max() <= 1.0:
            display_img = (display_img * 255).astype("uint8")
        else:
            display_img = display_img.astype("uint8")

        print(f"Image {idx} - Processed shape: {display_img.shape}, min: {display_img.min()}, max: {display_img.max()}")

        plt.imshow(display_img)
        plt.title(caption)
        plt.axis('off')
        plt.show()

