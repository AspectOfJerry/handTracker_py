import cv2
import mediapipe


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.55, trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1,  self.detectionConfidence, self.trackingConfidence)
        self.mediapipeDraw = mediapipe.solutions.drawing_utils

    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if (self.results.multi_hand_landmarks):
            for handLandmarks in self.results.multi_hand_landmarks:
                if (draw):
                    cv2.putText(image, "Confidence: " + str(self.results.multi_handedness[0].classification[0].score) or "ERR", (20, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (253, 253, 253), 1)
                    self.mediapipeDraw.draw_landmarks(image, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return image

    def getPos(self, image, handN=0, draw=True):
        landmarks = []

        if (self.results.multi_hand_landmarks):
            hand = self.results.multi_hand_landmarks[handN]

            for id, landmark in enumerate(hand.landmark):
                height, width, channel = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # print(id, cx, cy)
                landmarks.append([id, cx, cy])
                if (draw):
                    cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # img, center, radius, color, thickness

        return landmarks
