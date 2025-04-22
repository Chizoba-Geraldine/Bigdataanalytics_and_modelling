# Import required libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Set paths to the dataset (update this path to your local directory)
DATASET_PATH = "./PetImages"
CATEGORIES = ["Cat", "Dog"]

# Function to load and preprocess images
def load_data(img_size=64):
    data = []
    labels = []
    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        class_label = CATEGORIES.index(category)  # Cat=0, Dog=1
        for img_name in tqdm(os.listdir(path), desc=f"Processing {category} images"):
            try:
                # Read image in grayscale, resize, and normalize
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                data.append(img.flatten())  # Flatten the image for SVM
                labels.append(class_label)
            except Exception as e:
                # Handle any unreadable/corrupt images
                pass
    return np.array(data), np.array(labels)

# Load and preprocess the dataset
print("Loading dataset...")
IMG_SIZE = 64  # Resize images to 64x64
X, y = load_data(IMG_SIZE)
print(f"Dataset loaded with {len(X)} samples.")

# Split the data into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
print("Training the SVM model...")
model = SVC(kernel='linear', C=1.0, random_state=42)  # Linear kernel for simplicity
model.fit(X_train, y_train)

# Make predictions
print("Evaluating the model...")
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
