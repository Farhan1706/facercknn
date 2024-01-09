import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import csv
from datetime import datetime

# Inisialisasi KNN dan load model
facecascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
facecascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_profileface.xml')

# Mendapatkan daftar file di dalam folder /data
folder_path = 'data/'
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
knn = KNeighborsClassifier(n_neighbors=3)

# Mengumpulkan data wajah dan nama dari semua pengguna
all_faces = []
all_names = []
for user_data in faces_and_names.values():
    all_faces.extend(user_data['faces'])
    all_names.extend(user_data['names'])

knn.fit(all_faces, all_names)

# Inisialisasi camera
camera = cv2.VideoCapture(2)

# Inisialisasi CSV
csv_filename = 'kehadiran.csv'
csv_exists = os.path.exists(csv_filename)

fieldnames = ['Nama', 'Waktu']

# Variabel untuk melacak apakah data sudah ditambahkan ke CSV
known_names = set(all_names)  # Mengubah list nama menjadi set untuk pemrosesan yang lebih efisien
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

            # Menetapkan label "Tidak Diketahui" jika nilai presisi kurang dari 0.8
            if confidence < 0.8:
                label = 'Tidak Diketahui'
            else:
                label = f'{text[0]} ({confidence:.2f})'

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

            cv2.putText(frame, label, (NxHorizontal, NyVertikal-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (NxHorizontal, NyVertikal), (NxHorizontal + lebarPembatas, NyVertikal + tinggiPembatas), (0, 0, 255), 2)

        for (NxHorizontal, NyVertikal, lebarPembatas, tinggiPembatas) in face_coordinates_profile:
            fc = frame[NyVertikal:NyVertikal + tinggiPembatas, NxHorizontal:NxHorizontal + lebarPembatas, :]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            text = knn.predict(r)

            # Mendapatkan nilai confidence
            confidence = knn.predict_proba(r).max()

            # Menetapkan label "Tidak Diketahui" jika nilai presisi kurang dari 0.8
            if confidence < 0.8:
                label = 'Tidak Diketahui'
            else:
                label = f'{text[0]} ({confidence:.2f})'

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

            cv2.putText(frame, label, (NxHorizontal, NyVertikal-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (NxHorizontal, NyVertikal), (NxHorizontal + lebarPembatas, NyVertikal + tinggiPembatas), (0, 0, 255), 2)

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