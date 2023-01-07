
import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils

from pathlib import Path
import os
import ntpath

path="C:\\Users\\Hp elite book pro\\Desktop\\easy_facial_recognition\\"
print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
pose_predictor_68_point = dlib.shape_predictor(path+"pretrained_model\\shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor(path+"pretrained_model\\shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(path+"pretrained_model\\dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Importing pretrained model..')
print("=============")

def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    face_locations = face_detector(image, 1)#liste des coordonnes
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        #compute_face_descriptor=> va calculer un vecteur de 128 deux dimension qui va decrire en fait votre visage
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape)#recuperer les 68 points qui permet de d'ecrire le visage sons forme de coordinnes pour pouvoir les afficher
        landmarks_list.append(shape)
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list#les images coder et les points et liste ds coordonnes ou bien image reels et la copier de cette image avec des points ( le visage seule )


def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)


if __name__ == '__main__':

    print('[INFO] Importing faces...')
    face_to_encode_path = Path(path)    # le chemin des images contenant des faces a coder
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]  # filter les fichiers ayant comme extension jpg

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)
    if len(files)==0:#La fonction len() renvoie le nombre des éléments (ou la longueur) dans un objet
        raise ValueError('No faces detect in the directory: {}'.format(face_to_encode_path))
    known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]

    known_face_encodings = []
    for file_ in files:
        image = PIL.Image.open(file_)# ouvrir les images
        image = np.array(image)# je vais le mettre les passers en arrét donc en vecteur
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)


    print('[INFO] Faces well imported')
    print('[INFO] Starting Webcam...')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam well started')
    print('[INFO] Detecting...')
    while True:
        ret, frame = video_capture.read()
        easy_face_reco(frame, known_face_encodings, known_face_names)
        cv2.imshow('Easy Facial Recognition App', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Stopping System')
    video_capture.release()
    cv2.destroyAllWindows()

