import mediapipe as mp
import cv2

mpHands = mp.solutions.hands
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

class HandAndFaceDetection:
    def __init__(self, max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        self.face_detection = mpFaceDetection.FaceDetection(min_detection_confidence=min_detection_confidence)

    def findHandLandMarks(self, image, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        landMarkList = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landMarks = []
                for id, landMark in enumerate(hand_landmarks.landmark):
                    imgH, imgW, imgC = originalImage.shape
                    xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                    landMarks.append([id, xPos, yPos])
                landMarkList.append(landMarks)

            if draw:
                for hand_landmarks, landmarks in zip(results.multi_hand_landmarks, landMarkList):
                    mpDraw.draw_landmarks(originalImage, hand_landmarks, mpHands.HAND_CONNECTIONS)

        return originalImage, landMarkList

    def findFaces(self, image, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)
        faceList = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = originalImage.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                faceList.append([bbox, detection.score])

                if draw:
                    cv2.rectangle(originalImage, bbox, (255, 0, 0), 1)
                    cv2.putText(originalImage, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

        return originalImage, faceList