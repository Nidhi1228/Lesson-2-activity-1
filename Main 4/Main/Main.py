import mediapipe as mp
import cv2
import numpy as np



# Initialize MediaPipe Hands

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils


# Webcam setup

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():

    print("Error: Could not access the webcam.")

    exit()


while True:

    success, img = cap.read()

    if not success:

        print("Failed to read frame from webcam.")

        break


    img = cv2.flip(img, 1)  # Flip the image for a mirror effect

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks and results.multi_handedness:

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

            hand_label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'


            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)


# Show the video feed with annotations

    cv2.imshow("Gesture Volume and Brightness Controller", img)


    # Break the loop if 'q' is pressed

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


# Release the webcam and close all windows

cap.release()

cv2.destroyAllWindows()