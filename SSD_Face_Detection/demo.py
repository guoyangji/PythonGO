import cv2 as cv
from cv2 import dnn

net = dnn.readNetFromCaffe('./data/face_detector/deploy.prototxt',
                           './data/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel')

inWidth = 300
inHeight = 300
confThreshold = 0.5

img = cv.imread('test.png')
cols = img.shape[1]
rows = img.shape[0]

net.setInput(dnn.blobFromImage(img, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
detections = net.forward()
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confThreshold:
        xLeftBottom = int(detections[0, 0, i, 3] * cols)
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop = int(detections[0, 0, i, 5] * cols)
        yRightTop = int(detections[0, 0, i, 6] * rows)
        cv.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0),2)
cv.imshow("result", img)
cv.waitKey(0)
cv.destroyAllWindows()







