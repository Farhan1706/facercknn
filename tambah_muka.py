import os
import numpy as np
import cv2
import pickle

face_data = []
i = 0

camera = cv2.VideoCapture(2)

facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

name = input('Masukan Nama : ')
ret = True

while(ret):
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)

        for (a, b, w, h) in face_coordinates:
            faces = frame[b:b+h, a:a+w, :]
            resized_faces = cv2.resize(faces, (50, 50))

            if i % 10 == 0 and len(face_data) < 100:
                face_data.append(resized_faces)
            cv2.rectangle(frame, (a, b), (a+w, b+h), (255, 0, 0), 2)
        i += 1

        cv2.imshow('frames', frame)
        print(f"Jumlah data yang diambil: {len(face_data)}")

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break

        if cv2.waitKey(1) == 27 or len(face_data) >= 100:
            break
    else:
        print('error')
        break

cv2.destroyAllWindows()
camera.release()

face_data = np.asarray(face_data)
face_data = face_data.reshape(len(face_data), -1)

folder_path = 'data/'
name_filename = f'nama_{name}.pkl'
faces_filename = f'muka_{name}.pkl'

if name_filename not in os.listdir(folder_path):
    names = [name] * len(face_data)
    with open(os.path.join(folder_path, name_filename), 'wb') as file:
        pickle.dump(names, file)
else:
    with open(os.path.join(folder_path, name_filename), 'rb') as file:
        names = pickle.load(file)

    names = names + [name] * len(face_data)
    with open(os.path.join(folder_path, name_filename), 'wb') as file:
        pickle.dump(names, file)

if faces_filename not in os.listdir(folder_path):
    with open(os.path.join(folder_path, faces_filename), 'wb') as w:
        pickle.dump(face_data, w)
else:
    with open(os.path.join(folder_path, faces_filename), 'rb') as w:
        faces = pickle.load(w)

    faces = np.append(faces, face_data, axis=0)
    with open(os.path.join(folder_path, faces_filename), 'wb') as w:
        pickle.dump(faces, w)
