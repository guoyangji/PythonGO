
import cv2 as cv
from cv2 import dnn

net = dnn.readNetFromCaffe('./data/face_detector/deploy.prototxt',
                           './data/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel')

inWidth = 300
inHeight = 300
confThreshold = 0.5
num = 0
total_faces = 0
cap = cv.VideoCapture(0)
fps = int(cap.get(cv.CAP_PROP_FPS))
while True:
    faces = 0
    ret, frame = cap.read()
    num += 1
    if num % fps == 0:
        num = 0
        net.setInput(dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                faces += 1
        total_faces += faces
        print('real time faces : ',faces,'total_faces : ',total_faces)





