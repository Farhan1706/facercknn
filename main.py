import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import os
import csv
from datetime import datetime

# Inisialisasi KNN dan load model
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mendapatkan daftar file di dalam folder /data
folder_path = 'data/'
user_files = os.listdir(folder_path)

faces = []
names = []

# Membaca data wajah dan nama dari setiap file yang sesuai pola
for file_name in user_files:
    if file_name.startswith('muka_') and file_name.endswith('.pkl'):
        user_name = file_name.split('_')[1].split('.')[0]

        # Membaca data wajah
        muka_file_path = os.path.join(folder_path, file_name)
        with open(muka_file_path, 'rb') as fileMuka:
            user_faces = pickle.load(fileMuka)
            faces.extend(user_faces)

        # Membaca data nama
        nama_file_name = 'nama_' + user_name + '.pkl'
        nama_file_path = os.path.join(folder_path, nama_file_name)
        with open(nama_file_path, 'rb') as fileNama:
            user_names = pickle.load(fileNama)
            names.extend(user_names)

# Inisialisasi KNN
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(faces, names)

# Inisialisasi camera
camera = cv2.VideoCapture(2)

# Inisialisasi CSV
csv_filename = 'kehadiran.csv'
csv_exists = os.path.exists(csv_filename)

with open(csv_filename, 'a', newline='') as csv_file:
    fieldnames = ['Nama', 'Waktu']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    if not csv_exists:
        writer.writeheader()

# Variabel untuk melacak apakah data sudah ditambahkan ke CSV
data_added = False
known_names = set(names)  # Mengubah list nama menjadi set untuk pemrosesan yang lebih efisien
current_name = None

while True:
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)

        for (a, b, w, h) in face_coordinates:
            fc = frame[b:b + h, a:a + w, :]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            text = knn.predict(r)
            cv2.putText(frame, text[0], (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (a, b), (a + w, b + h), (0, 0, 255), 2)

            # Menyimpan data kehadiran ke dalam file CSV hanya jika data belum ditambahkan
            if text[0] in known_names:
                if text[0] != current_name:
                    current_name = text[0]
                    data_added = False  # Reset data_added jika nama berbeda

                if not data_added:
                    with open(csv_filename, 'a', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow({'Nama': text[0], 'Waktu': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                    data_added = True

        cv2.imshow('Sistem Pengenalan Wajah', frame)

        # Menambahkan logika untuk menghentikan program OpenCV jika tombol 'q' ditekan
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    else:
        print("error")
        break

cv2.destroyAllWindows()
camera.release()