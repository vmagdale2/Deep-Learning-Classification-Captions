# Deep-Learning-Classification-Captions
Deep Learning for Image Classification and Captioning: A Comprehensive Project

# ✈️ Aircraft Damage Classification & Captioning

This deep learning project aims to build an end-to-end system that classifies the extent of damage on aircraft images and generates descriptive captions to support inspection processes.

---

## 🧠 Project Objectives
- Classify aircraft images into "Minor Damage" and "Severe Damage"
- Fine-tune a CNN model for accurate classification
- Generate image captions based on model predictions
- Visualize predictions for human-in-the-loop inspection
- Evaluate and compare model performance over multiple iterations

---

## 📂 Dataset Overview
- **Source**: Provided `aircraft-damage-dataset-v1`
- **Classes**: Minor Damage (0), Severe Damage (1)
- **Preprocessing**:
  - Images resized to `224x224`
  - Normalized pixel values
  - Split into training, validation, and test sets

---

## 🏗️ Modeling Workflow

### ✅ 01 - Data Preprocessing
- Loaded, resized, and scaled image data
- Converted labels to one-hot format
- Saved preprocessed `.npy` files for fast reuse

### ✅ 02 - Model Training
- Built multiple CNN architectures with:
  - Conv2D, MaxPooling, Dropout, and Dense layers
  - L2 regularization and callbacks (EarlyStopping, ReduceLROnPlateau)
- Selected **Model 3** after testing various configurations
- Saved as `model_3.keras`

### ✅ 03 - Model Evaluation
- Plotted training vs. validation accuracy and loss
- Generated classification reports and confusion matrices
- Compared base vs. retrained (re-weighted) model
- **Model 3 achieved 76% accuracy with balanced performance**

### ✅ 04 - Image Captioning
- Applied predictions on test images
- Mapped labels to human-readable captions
- Ensured display formatting and grayscale normalization

### ✅ 05 - Final Inference
- Loaded Model 3 and test images
- Generated and displayed top 10 captioned predictions
- Saved all predictions (with confidence scores) to `final_predictions.txt`

---

## 📊 Final Model Performance (Model 3)
- **Accuracy**: 0.76
- **Precision**: 0.76
- **Recall**: 0.76
- **F1 Score**: 0.76

Confusion Matrix:
```
Actual \ Predicted  | Class 0 | Class 1
---------------------|---------|---------
Class 0              | 27      | 7
Class 1              | 10      | 26
```

---

## 🖼️ Sample Output

```
Prediction: Severe Damage (71.56%)
```

---

## 💡 Key Learnings
- Avoid excessive class weighting — can over-bias predictions
- L2 regularization + dropout helped avoid overfitting
- One-hot encoding + softmax ensured correct label learning
- Direct image captioning helped verify prediction quality

---

## 📁 Repo Structure
```
├── notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_image_captioning.ipynb
│   ├── 05_final_inference.ipynb
│   └── readme.md
│
├── scripts
│   ├── data_preprocessing.py
│   ├── final_inference.py
│   ├── image_captioning.py
│   ├── model_evaluation.py
│   └── model_training.py
│
├── utils
│   └── utils.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── final_predictions.pdf
```

---

**Made ️ by Veronica Magdaleno**




