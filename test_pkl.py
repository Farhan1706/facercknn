import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

# Inisialisasi KNN dan load model
facecascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
facecascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_profileface.xml')

# Mendapatkan daftar file di dalam folder /data
folder_path = 'data'
user_files = os.listdir(folder_path)

faces_and_names = {}  # Dictionary untuk menyimpan data wajah dan nama

# Membaca data wajah dan nama dari setiap file yang sesuai pola
for file_name in user_files:
    if file_name.endswith('.pkl'):
        user_name = file_name.split('.')[0]

        # Membaca data wajah dan nama
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            try:
                data = pickle.load(file)

                # Memeriksa apakah data adalah dictionary
                if isinstance(data, dict):
                    user_data = {'faces': data.get('faces', []), 'names': data.get('names', [])}
                    faces_and_names[user_name] = user_data
                else:
                    print(f"Skipping file {file_name} - Invalid data structure.")

            except Exception as e:
                print(f"Error loading data from file {file_name}: {str(e)}")

# Inisialisasi KNN
knn = KNeighborsClassifier(n_neighbors=20)

# Mengumpulkan data wajah dan nama dari semua pengguna
all_faces = []
all_names = []
for user_data in faces_and_names.values():
    all_faces.extend(user_data['faces'])
    all_names.extend(user_data['names'])

# Split data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(all_faces, all_names, test_size=0.2, random_state=42)

# Fit KNN model menggunakan train set
knn.fit(X_train, y_train)

# Predict menggunakan test set
y_pred = knn.predict(X_test)

# Menghitung dan menampilkan akurasi
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2%}")

# Menghitung dan menampilkan confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Menampilkan classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)