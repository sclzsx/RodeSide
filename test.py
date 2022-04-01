import cv2
import numpy as np
bgr = cv2.imread('Data/5.JPG')
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
print(hsv[:,:,0])