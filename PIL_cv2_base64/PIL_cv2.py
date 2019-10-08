
import cv2
from PIL import Image
import numpy as np
def PIL_2_cv2(img_path):
    image = Image.open(img_path)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img


import cv2
from PIL import Image
def cv2_2_PIL(img_path):
    image = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return img






