import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import time

image = cv2.imread('images/gus.jpg')

# Convert the image to RGB colorspace
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 4, 6)

# for (x, y, w, h) in faces:
#     cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 3)
#
# image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Display the image
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Original Image')
ax1.imshow(gray)

plt.show()


