import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import os
import csv
from datetime import datetime

# Inisialisasi KNN dan load model
facecascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
facecascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_profileface.xml')

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

        # Deteksi wajah frontal
        face_coordinates_frontal = facecascade_frontal.detectMultiScale(gray, 1.1, 5)
        # Deteksi wajah profil
        face_coordinates_profile = facecascade_profile.detectMultiScale(gray, 1.1, 5)

        for (NxHorizontal, NyVertikal, lebarPembatas, tinggiPembatas) in face_coordinates_frontal:
            fc = frame[NyVertikal:NyVertikal + tinggiPembatas, NxHorizontal:NxHorizontal + lebarPembatas, :]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            text = knn.predict(r)

            # Mendapatkan nilai confidence
            confidence = knn.predict_proba(r).max()

            # Menampilkan nama dan confidence pada frame
            label = f'{text[0]} ({confidence:.2f})'
            cv2.putText(frame, label, (NxHorizontal, NyVertikal-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (NxHorizontal, NyVertikal), (NxHorizontal + lebarPembatas, NyVertikal + tinggiPembatas), (0, 0, 255), 2)

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

        for (NxHorizontal, NyVertikal, lebarPembatas, tinggiPembatas) in face_coordinates_profile:
            fc = frame[NyVertikal:NyVertikal + tinggiPembatas, NxHorizontal:NxHorizontal + lebarPembatas, :]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            text = knn.predict(r)

            # Mendapatkan nilai confidence
            confidence = knn.predict_proba(r).max()

            # Menampilkan nama dan confidence pada frame
            label = f'{text[0]} ({confidence:.2f})'
            cv2.putText(frame, label, (NxHorizontal, NyVertikal-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (NxHorizontal, NyVertikal), (NxHorizontal + lebarPembatas, NyVertikal + tinggiPembatas), (0, 0, 255), 2)

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
