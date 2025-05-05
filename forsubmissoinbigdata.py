import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
dataset_path = "C:\\Users\\Student\\Desktop\\bd\\Dataset_BUSI_with_GT"

def load_dataset(dataset_path, img_size=(128, 128)):
    images, labels = [], []
    categories = ['benign', 'malignant', 'normal']
    for category in categories:
        image_paths = glob(os.path.join(dataset_path, category, f"{category} (*).png"))
        image_paths = [p for p in image_paths if not p.endswith('_mask.png')]
        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(category)
    return np.array(images), np.array(labels)

# Load and preprocess data
IMG_SIZE = (128, 128)
images, labels = load_dataset(dataset_path, IMG_SIZE)
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
class_names = le.classes_

X_ml = images.reshape(len(images), -1) / 255.0
X_cnn = images[..., np.newaxis] / 255.0

X_train_ml, X_test_ml, y_train, y_test = train_test_split(
    X_ml, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train_cnn, X_test_cnn = (
    X_train_ml.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1),
    X_test_ml.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
)

def evaluate_model(model_name, y_true, y_pred, class_names):
    print(f"\n--- {model_name} ---")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Classical ML Models
print("\nTraining SVM...")
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_ml, y_train)
evaluate_model("SVM", y_test, svm.predict(X_test_ml), class_names)

print("\nTraining KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_ml, y_train)
evaluate_model("KNN", y_test, knn.predict(X_test_ml), class_names)

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_ml, y_train)
evaluate_model("Random Forest", y_test, rf.predict(X_test_ml), class_names)

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_ml, y_train)
evaluate_model("Logistic Regression", y_test, lr.predict(X_test_ml), class_names)

# CNN Model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

cnn_model = create_cnn_model((128, 128, 1), len(class_names))
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test_cnn, y_test),
    verbose=1
)

# Plot Validation Accuracy vs Epoch
plt.figure(figsize=(8, 5))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue', marker='o')
plt.title("CNN Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate CNN
y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
evaluate_model("CNN", y_test, y_pred_cnn, class_names)
