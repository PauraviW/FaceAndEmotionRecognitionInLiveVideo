import cv2
import numpy
from utils import helper as helper

parent_directory = 'database'

images, labels = helper.generate_dataset(parent_directory)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
model.save('..\\trained_models\\face_recognition.xml')
print("saved")
