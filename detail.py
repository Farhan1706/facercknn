import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle
import numpy as np
import os
import csv
from datetime import datetime

# ... (Your existing code)

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# ... (Your existing code)

while True:
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ... (Your existing code)

        for (NxHorizontal, NyVertikal, lebarPembatas, tinggiPembatas) in face_coordinates_frontal:
            # ... (Your existing code)

            # Append true label and predicted label to lists
            true_labels.append(user_name)
            predicted_labels.append(text[0])

            # ... (Your existing code)

        for (NxHorizontal, NyVertikal, lebarPembatas, tinggiPembatas) in face_coordinates_profile:
            # ... (Your existing code)

            # Append true label and predicted label to lists
            true_labels.append(user_name)
            predicted_labels.append(text[0])

            # ... (Your existing code)

        cv2.imshow('Sistem Pengenalan Wajah', frame)

        # ... (Your existing code)

    else:
        print("error")
        break

# Calculate precision, recall, f1-score, and support
precision, recall, fscore, support = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

# Print the results
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {fscore:.2f}')
print(f'Support: {support}')

cv2.destroyAllWindows()
camera.release()
