
import base64
import cv2
def cv2_2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str



import base64
import numpy as np
import cv2
def base64_2_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparray = np.fromstring(imgString, np.uint8)
    image = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    return image







