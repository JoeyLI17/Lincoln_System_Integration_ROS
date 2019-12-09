# from light_classification.tl_classifier import TLClassifier
import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)
cv2.imwrite('origin.png',img)

# Draw a diagonal blue line with thickness of 5 px bgr
# img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),5)

cv2.imwrite('text.jpg',img)