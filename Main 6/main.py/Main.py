import cv2
import mediapipe as mp
import time
import numpy as np


# Initialize MediaPipe Hands

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


# Webcam setup
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

if not cap.isOpened():

    print("Error: Could not access the webcam.")
    exit()


# Timestamp for debouncing gestures

last_action_time = 0

debounce_time = 1  # 1 second debounce between actions


while True:

    success, img = cap.read()

    if not success:

        print("Failed to read frame from webcam.")

        break


    img = cv2.flip(img, 1)  # Flip the image for a mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)


    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks,hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
             
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_label = hand_info.classification[0].label  # 'Left' or 'Right'
             
            # Get key landmarks

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]


            # Frame dimensions

            frame_height, frame_width, _ = img.shape


            # Convert normalized coordinates to pixel coordinates

            thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
            middle_x, middle_y = int(middle_tip.x * frame_width), int(middle_tip.y * frame_height)
            ring_x, ring_y = int(ring_tip.x * frame_width), int(ring_tip.y * frame_height)
            pinky_x, pinky_y = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)


            # Draw circles for landmarks

            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)

            cv2.circle(img, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)

            cv2.circle(img, (middle_x, middle_y), 10, (0, 0, 255), cv2.FILLED)

            cv2.circle(img, (ring_x, ring_y), 10, (255, 255, 0), cv2.FILLED)

            cv2.circle(img, (pinky_x, pinky_y), 10, (255, 0, 255), cv2.FILLED)


            # Gesture Logic

            current_time = time.time()


            # Click picture: Thumb touches Index finger

            if abs(thumb_x - index_x) < 30 and abs(thumb_y - index_y) < 30:

                if current_time - last_action_time > debounce_time:

                    cv2.putText(img, "Picture Captured!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Hand - {hand_label}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    last_action_time = current_time

                    cv2.imwrite(f"picture_{int(time.time())}.jpg", img)

                    print("Picture saved!")




    cv2.imshow("Gesture-Controlled Photo App", img)



    # Exit on 'q'

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


# Release resources

cap.release()

cv2.destroyAllWindows()