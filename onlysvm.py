# svm_cats_dogs_simple.py

import os
import time
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from joblib import dump

# ==== 1. Set dataset paths (update to match your folders) ====
TRAIN_DIR = "./training_set_1/"
TEST_DIR = "./test_set_1/"

# ==== 2. Load images and labels as raw pixel data ====
IMG_SIZE = (64, 64)  # You can increase this, but keep it reasonable for performance

def load_images_and_labels(base_dir):
    images = []
    labels = []
    for label in ["cats", "dogs"]:
        folder = os.path.join(base_dir, label)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                with Image.open(fpath).convert("RGB") as img:
                    img = img.resize(IMG_SIZE)
                    arr = np.array(img).flatten() / 255.0  # Normalize pixel values
                    images.append(arr)
                    labels.append(0 if label == "cats" else 1)
            except Exception as e:
                print(f"[WARNING] Skipping file {fpath}: {e}")
    return np.array(images), np.array(labels)

print("[INFO] Loading training images...")
X_train, y_train = load_images_and_labels(TRAIN_DIR)

print("[INFO] Loading test images...")
X_test, y_test = load_images_and_labels(TEST_DIR)

# ==== 3. Train SVM model ====
svm = SVC(kernel='rbf', C=1.0, gamma='scale')

print("[INFO] Training SVM...")
start_train = time.time()
svm.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"[INFO] Training time: {train_time:.2f} seconds")

# ==== 4. Save the trained model ====
dump(svm, "svm_model_raw_pixels.joblib")

# ==== 5. Evaluate model ====
print("[INFO] Evaluating model...")
start_test = time.time()
y_pred = svm.predict(X_test)
test_time = time.time() - start_test
print(f"[INFO] Testing time: {test_time:.2f} seconds")

# ==== 6. Print evaluation metrics ====
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["cats", "dogs"]))
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Training Time: {train_time:.2f} seconds")
print(f"Testing Time: {test_time:.2f} seconds")
