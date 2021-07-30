import tensorflow.keras as tf
import numpy as np
import cv2
import smtplib
from passgmail import password
import os
import mediapipe as mp
import face_recognition


cap = cv2.VideoCapture(0)
face_detector = mp.solutions.face_detection
face_detection = face_detector.FaceDetection()
np.set_printoptions(suppress=True)
model = tf.models.load_model("Face Mask detector.h5")
condition = "None"
color = (255, 0, 0)
p_name = ""


img_directory = "Images"
lst = os.listdir(img_directory)
imgs = []
class_names = []
for cl in lst:
    img = cv2.imread(os.path.join(img_directory, cl))
    imgs.append(img)
    class_names.append(os.path.splitext(cl)[0])


def find_encodings(imgs):
    encode_lst = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_lst.append(encode)
    return encode_lst

encode_lst = find_encodings(imgs)


def send_email(email, name):
    global p_name
    if name != p_name:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('adk.dipin@gmail.com', password)
        server.sendmail('adk.dipin@gmail.com',
                        email,
                        f'Hey {name} you need to wear a mask in public place.')
        p_name = name

def face_detector(img):
    email_lst = ["dibyachitwan@gmail.com", "adhikari.dipin2@gmail.com", "dipin.adk@gmail.com"]
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faces_location = face_recognition.face_locations(imgs)
    encode_faces = face_recognition.face_encodings(imgs, faces_location)

    for encode_face in faces_location, encode_faces:
        results = face_recognition.compare_faces(encode_lst, encode_face)
        distance = face_recognition.face_distance(encode_lst, encode_face)

        max_class = np.argmin(distance)
        person = class_names[max_class].upper()
        email = email_lst[max_class]
        send_email(email, person)

        

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bbox_class = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), int(bbox_class.width * w), int(bbox_class.height * h)

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
            cropped_img = img[bbox[1] -50:bbox[1] + bbox[3] + 50, bbox[0] - 50:bbox[0] + bbox[2] + 50].copy()
            resized_img = cv2.resize(cropped_img, (224, 224))

            img_array = np.asarray(resized_img)
            normalized_img_array = (img_array.astype(np.float32) / 127 - 1)
            data[0] = normalized_img_array

            prediction = model.predict(data)
            max_number = max(prediction[0]) * 100
            max_class = np.argmax(prediction[0])
            if max_number >= 60:
                if max_class == 0:
                    condition = "Face Mask"
                    color = (0, 255, 0)
                    org = (bbox[0] - 50, bbox[1] - 10)
                else:
                    condition = "Without Face Mask" 
                    color = (0, 0, 255)
                    org = (bbox[0] - 100, bbox[1] - 10)
                    face_detector(resized_img)
            else:
                condition = "None"
                color = (255, 0, 0)
                org = (bbox[0], bbox[1] - 10)
            condition = condition + '-' + str(int(max_number)) + '%'

            img = cv2.rectangle(img, bbox, color, 3)
            cv2.putText(img, condition, org, cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    cv2.imshow('Image', img)
    bbox = []
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()