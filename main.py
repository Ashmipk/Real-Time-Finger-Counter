import cv2 as cv
import mediapipe as mp

# Setup mediapipe
mpHands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

# Create the Hands object
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# GET HAND LANDMARKS
def getHandlandmarks(img, draw=True):
    allHands = []
    frameRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    handsDetected = hands.process(frameRGB)

    if handsDetected.multi_hand_landmarks:
        for idx, landmark in enumerate(handsDetected.multi_hand_landmarks):
            lmlist = []
            h, w, c = img.shape
            for id, lm in enumerate(landmark.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((id, cx, cy))

            # Hand label (Left/Right)
            handType = handsDetected.multi_handedness[idx].classification[0].label
            allHands.append((lmlist, handType))

            if draw:
                drawing.draw_landmarks(
                    image=img,
                    landmark_list=landmark,
                    connections=mpHands.HAND_CONNECTIONS
                )
    return allHands  # ✅ return list of (landmarks, handType)


# FINGER COUNT
def fingerCount(lmlist, handType):
    count = 0
    if lmlist[8][2] < lmlist[6][2]:   # Index
        count += 1
    if lmlist[12][2] < lmlist[10][2]: # Middle
        count += 1
    if lmlist[16][2] < lmlist[14][2]: # Ring
        count += 1
    if lmlist[20][2] < lmlist[18][2]: # Pinky
        count += 1

    # ✅ Thumb depends on left vs right hand
    if handType == "Right":
        if lmlist[4][1] < lmlist[2][1]:
            count += 1
    else:  # Left hand
        if lmlist[4][1] > lmlist[2][1]:
            count += 1

    return count


# CAMERA
cam = cv.VideoCapture(0)

while True:
    success, frame = cam.read()
    if not success:
        print("Camera not detected")
        break

    frame = cv.flip(frame, 1)

    allHands = getHandlandmarks(img=frame, draw=True)
    totalFingers = 0
    for lmlist, handType in allHands:   # ✅ get both landmarks + type
        totalFingers += fingerCount(lmlist, handType)

    if totalFingers > 0:
        # cv.rectangle(frame, (400, 10), (600, 250), (0, 0, 0), -1)
        cv.putText(frame, str(totalFingers), (400, 250),
                   cv.FONT_HERSHEY_PLAIN, 12, (0, 255, 255), 30)

    cv.imshow("Hand Landmark", frame)

    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
