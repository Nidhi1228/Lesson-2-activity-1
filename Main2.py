import cv2

import numpy as np

import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():

 raise RuntimeError("Could not open camera")

# print(frame.shape, frame.dtype) # e.g., (480, 640, 3) uint8

while True:


 ret, frame = cap.read()

############## Skin Masking #############

# 1. Convert to HSV for color filtering

 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 2. Define the range for skin color in HSV

 lower_skin = np.array([0, 20, 70], dtype=np.uint8)

 upper_skin = np.array([40, 255, 255], dtype=np.uint8)


# 3. Create a mask to detect skin color

 mask = cv2.inRange(hsv, lower_skin, upper_skin)

# 4. Apply the mask to the frame

 result = cv2.bitwise_and(frame, frame, mask=mask)

############# Applying Contours #############

 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If contours are found, draw them

 if contours:

# print("Flag 1")

  max_contour = max(contours, key=cv2.contourArea) # Get largest contour

 if cv2.contourArea(max_contour) > 500: # Ignore small contours

# Draw the bounding box around the detected hand

  x, y, w, h = cv2.boundingRect(max_contour)

  cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Get the center of the hand for further tracking or interaction

  center_x = int(x + w / 2)

  center_y = int(y + h / 2)

  cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1) # Red dot at cente

  cv2.imshow("original Frame",frame)

  cv2.imshow("Filtered Frame", result)

# Exit on q / ESC

  key = cv2.waitKey(1) & 0xFF

 if key in (ord('q'), 27):

  break


# cv2.waitkey(0)

cap.release()

cv2.destroyAllWindows()