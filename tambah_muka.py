import os
import numpy as np
import cv2
import pickle

# Inisialisasi cascade classifier untuk deteksi wajah frontal dan profile
facecascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
facecascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Fungsi untuk mendeteksi wajah dan mendapatkan data wajah
def detect_and_extract_faces(frame, face_data, i):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah frontal
    face_coordinates_frontal = facecascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Deteksi wajah profile
    face_coordinates_profile = facecascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (a, b, w, h) in face_coordinates_frontal:
        faces = frame[b:b+h, a:a+w, :]
        resized_faces = cv2.resize(faces, (50, 50))

        if i % 10 == 0 and len(face_data) < 100:
            face_data.append(resized_faces)
            cv2.rectangle(frame, (a, b), (a+w, b+h), (255, 0, 0), 2)

    for (a, b, w, h) in face_coordinates_profile:
        faces = frame[b:b+h, a:a+w, :]
        resized_faces = cv2.resize(faces, (50, 50))

        if i % 10 == 0 and len(face_data) < 100:
            face_data.append(resized_faces)
            cv2.rectangle(frame, (a, b), (a+w, b+h), (255, 0, 0), 2)

    return face_data

face_data = []
i = 0

camera = cv2.VideoCapture(2)

name = input('Masukkan Nama: ')
ret = True

while ret:
    ret, frame = camera.read()
    if ret:
        face_data = detect_and_extract_faces(frame, face_data, i)
        i += 1

        cv2.imshow('frames', frame)
        print(f"Jumlah data yang diambil: {len(face_data)}")

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q') or len(face_data) >= 100:
            break
    else:
        print('error')
        break

cv2.destroyAllWindows()
camera.release()

face_data = np.asarray(face_data)
face_data = face_data.reshape(len(face_data), -1)

folder_path = 'data/'
file_name = f'{name}.pkl'

if file_name not in os.listdir(folder_path):
    data = {'names': [name] * len(face_data), 'faces': face_data}
    with open(os.path.join(folder_path, file_name), 'wb') as file:
        pickle.dump(data, file)
else:
    with open(os.path.join(folder_path, file_name), 'rb') as file:
        data = pickle.load(file)

    data['names'] += [name] * len(face_data)
    data['faces'] = np.append(data['faces'], face_data, axis=0)

    with open(os.path.join(folder_path, file_name), 'wb') as file:
        pickle.dump(data, file)
