# svm_cats_dogs_split.py

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from joblib import dump

# ==== 1. Set dataset paths (UPDATE THESE TO MATCH YOUR FILE SYSTEM) ====
TRAIN_DIR = "./training_set_1/"
TEST_DIR  = "./test_set_1/"
# ==== 2. Load image paths and labels ====
def load_image_paths_labels(base_dir):
    image_paths = []
    labels = []
    for label in ["cats", "dogs"]:
        folder = os.path.join(base_dir, label)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder, fname))
                labels.append(0 if label == "cats" else 1)
    return np.array(image_paths), np.array(labels)

train_paths, y_train = load_image_paths_labels(TRAIN_DIR)
test_paths, y_test   = load_image_paths_labels(TEST_DIR)

# ==== 3. Feature extractor using MobileNetV2 (pretrained) ====
IMG_SIZE = (224, 224)
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_features(paths):
    features = []
    for p in paths:
        try:
            img = load_img(p, target_size=IMG_SIZE)
            arr = img_to_array(img)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
            features.append(arr)
        except Exception as e:
            print(f"[WARNING] Skipping file {p}: {e}")
    features = np.array(features)
    return base_model.predict(features, batch_size=32, verbose=1)

print("[INFO] Extracting training features...")
X_train = extract_features(train_paths)

print("[INFO] Extracting test features...")
X_test = extract_features(test_paths)

# ==== 4. Train the SVM model ====
svm = SVC(kernel='rbf', C=1.0, gamma='scale')

print("[INFO] Training SVM...")
start_train = time.time()
svm.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"[INFO] Training time: {train_time:.2f} seconds")

# ==== 5. Save the model ====
dump(svm, "svm_model_cats_dogs.joblib")

# ==== 6. Evaluate the model ====
print("[INFO] Evaluating model...")
start_test = time.time()
y_pred = svm.predict(X_test)
test_time = time.time() - start_test
print(f"[INFO] Testing time: {test_time:.2f} seconds\n")

# ==== 7. Output evaluation metrics ====
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
