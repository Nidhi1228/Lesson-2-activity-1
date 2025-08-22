import cv2
# import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
  raise RuntimeError("Could not open camera")

# # # print(frame.shape, frame.dtype) # e.g., (480, 640, 3) uint8

while True:
 ret, frame = cap.read()
 cv2.imshow("original Frame",frame)

# # # Exit on q / ESC
 key = cv2.waitKey(1) & 0xFF
 if key in (ord('q'), 27):
  break

# cv2.waitkey(0)
cap.release()
cv2.destroyAllWindows()