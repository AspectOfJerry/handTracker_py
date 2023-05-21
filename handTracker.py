import cv2
import mediapipe


class handTracker():
    # leave `mode` True for video streams (false for stream of unrelated images), maxHands, detectionConfidence, trackingConfidence
    def __init__(self, mode=True, maxHands=2, detectionConfidence=0.60, trackingConfidence=0.55):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1,  self.detectionConfidence, self.trackingConfidence)
        self.mediapipeDraw = mediapipe.solutions.drawing_utils

    # image, draw on the image?
    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if (self.results.multi_hand_landmarks):
            for handLandmarks in self.results.multi_hand_landmarks:
                if (draw):
                    cv2.putText(image, f"Confidence: {str(round(self.results.multi_handedness[0].classification[0].score, 3))}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (253, 253, 253), 1)
                    self.mediapipeDraw.draw_landmarks(image, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return image

    # image, hand number, draw on the image?
    def getPos(self, image, handN=0, draw=True):
        landmarks = []

        if (self.results.multi_hand_landmarks):
            hand = self.results.multi_hand_landmarks[handN]

            for id, landmark in enumerate(hand.landmark):
                height, width, channel = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)

                landmarks.append([id, cx, cy])
                if (draw):
                    cv2.circle(image, (cx, cy), 5, (0, 255, 0), 1)  # img, center, radius, color, thickness

        return landmarks
