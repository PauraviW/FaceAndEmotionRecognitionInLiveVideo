import cv2
import sys
import os
from PIL import Image, ImageOps


class Database:
    def __init__(self):
        self.front_haar = "..\\haar_classifiers\\haarcascade_frontalface_default.xml"  # "haarcascade_profileface.xml"# 'haarcascade_frontalface_default.xml'
        self.side_haar = "..\\haar_classifiers\\haarcascade_profileface.xml"  # "haarcascade_profileface.xml"# 'haarcascade_frontalface_default.xml'
        self.image_width = 112
        self.image_height = 92
        self.size = 4

    def capture_photo(self, parent_database, label):
        path = os.path.join(parent_database, label)
        if not os.path.isdir(path):
            os.mkdir(path)

        front_cascade = cv2.CascadeClassifier(self.front_haar)
        side_cascade = cv2.CascadeClassifier(self.side_haar)

        webcam = cv2.VideoCapture(0)
        count = 0
        while count < 1000:
            (_, im) = webcam.read()
            im = cv2.flip(im, 1, 0)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray, (int(gray.shape[1] / self.size), int(gray.shape[0] / self.size)))

            if len(front_cascade.detectMultiScale(mini)) == 1 and len(side_cascade.detectMultiScale(mini)) == 0:
                faces = front_cascade.detectMultiScale(mini)
            elif len(front_cascade.detectMultiScale(mini)) == 0 and len(side_cascade.detectMultiScale(mini)) == 1:
                faces = side_cascade.detectMultiScale(mini)
            else:
                faces = front_cascade.detectMultiScale(mini)
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x, y, w, h) = [v * self.size for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (self.image_width, self.image_height))
                pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                              if n[0] != '.'] + [0])[-1] + 1
                cv2.imwrite('%s/%s.jpg' % (path, pin), face_resize)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(im, label, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0))
                imm = Image.open('%s/%s.jpg' % (path, pin))
                pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                              if n[0] != '.'] + [0])[-1] + 2
                im_mirror = ImageOps.mirror(imm)
                im_mirror.save('%s/%s.jpg' % (path, pin), quality=95)
                count += 1

            cv2.imshow('Capturing', im)
            key = cv2.waitKey(5)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("usage: python generate_database.py person_name")
    parent_database = 'database'
    if not os.path.isdir(parent_database):
        os.mkdir(parent_database)
    label = sys.argv[1]
    database = Database()
    database.capture_photo(parent_database, label)
