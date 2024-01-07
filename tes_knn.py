import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle


# Load the existing data
folder_path = 'data'
user_files = os.listdir(folder_path)

faces_and_names = {}  # Dictionary to store facial data and names

# Read facial data and names from each file
for file_name in user_files:
    if file_name.endswith('.pkl'):
        user_name = file_name.split('.')[0]

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            try:
                data = pickle.load(file)

                if isinstance(data, dict):
                    user_data = {'faces': data.get('faces', []), 'names': data.get('names', [])}
                    faces_and_names[user_name] = user_data
                else:
                    print(f"Skipping file {file_name} - Invalid data structure.")

            except Exception as e:
                print(f"Error loading data from file {file_name}: {str(e)}")

# Collect all facial data and names from all users
all_faces = []
all_names = []
for user_data in faces_and_names.values():
    all_faces.extend(user_data['faces'])
    all_names.extend(user_data['names'])

# Convert lists to numpy arrays
all_faces = np.asarray(all_faces)
all_names = np.asarray(all_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_faces, all_names, test_size=0.2, random_state=42)

# Define a range of n_neighbors values to try
neighbors_range = range(1, 21)

# Initialize variables to store the best results
best_accuracy = 0
best_n_neighbors = 0

# Iterate over different n_neighbors values
for n_neighbors in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    accuracy = knn.score(X_test, y_test)
    
    # Check if the current model is the best
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_neighbors = n_neighbors

# Print the best n_neighbors value
print(f"Best n_neighbors: {best_n_neighbors}, Best Accuracy: {best_accuracy}")