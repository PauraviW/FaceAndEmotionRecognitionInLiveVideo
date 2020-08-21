import cv2
from utils import helper as helper


haar_file = 'haar_classifiers\\haarcascade_frontalface_default.xml'
haar_file_side = 'haar_classifiers\\haarcascade_profileface.xml'

face_database_parent_directory = 'face_recognition\\database'

face_model = cv2.face.LBPHFaceRecognizer_create()
face_model.read('trained_models\\face_recognition.xml')

expr_database_parent_directory = 'expression_recognition\\database'

expr_model = cv2.face.LBPHFaceRecognizer_create()
# expr_model = cv2.face.FisherFaceRecognizer_create()
expr_model.read('trained_models\\expression_classification.xml')

# Create a list of images and a list of corresponding names
face_names = helper.generate_classification_labels(face_database_parent_directory)
expr_names = helper.generate_classification_labels(expr_database_parent_directory)

(image_width, image_height) = (112, 92)

def identify(faces, face_model, expr_model):
    # print("here")
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (image_width, image_height))

        # Try to recognize the face
        face_prediction = face_model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print('face_pred', face_prediction)
        if face_prediction[1] < 100:
            cv2.putText(im, '%s - %.0f' % (face_names[face_prediction[0]],
                                           face_prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        else:
            cv2.putText(im, 'not recognized', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # Try to classify the expression
        expr_prediction = expr_model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print('expr_pred', expr_prediction)
        if expr_prediction[1] < 100:
            cv2.putText(im, '%s - %.0f' % (expr_names[expr_prediction[0]],
                                           expr_prediction[1]), (x - 20, y - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0))
        else:
            cv2.putText(im, 'not recognized', (x - 20, y - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))


face_cascade = cv2.CascadeClassifier(haar_file)
side_face_cascade = cv2.CascadeClassifier(haar_file_side)
webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    side_faces = side_face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 1 and len(side_faces) == 0:
        identify(faces, face_model, expr_model)
        pass
    elif len(faces) == 0 and len(side_faces) == 1:
        identify(side_faces, face_model, expr_model)
        pass
    elif len(faces) == 1 and len(side_faces) == 1:
        identify(faces, face_model, expr_model)
        pass
    else:
        immm = cv2.flip(im, 1)
        gray = cv2.cvtColor(immm, cv2.COLOR_BGR2GRAY)
        # print(gray)
        side_faces = side_face_cascade.detectMultiScale(gray, 1.3, 5)
        identify(side_faces, face_model, expr_model)

    cv2.imshow('Integrated Models', im)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
