import cv2

import numpy as np

# import matplotlib.pyplot as plt

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

 cv2.imshow("original Frame",frame)

 cv2.imshow("Filtered Frame", result)

# Exit on q / ESC

 key = cv2.waitKey(1) & 0xFF

 if key in (ord('q'), 27):

  break


# cv2.waitkey(0)

cap.release()

cv2.destroyAllWindows()