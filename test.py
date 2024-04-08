from handDetection import HandAndFaceDetection
import cv2

detector = HandAndFaceDetection(min_detection_confidence=0.5, min_tracking_confidence=0.5)
webcam = cv2.VideoCapture(0)

while True:
    success, frame = webcam.read()
    if not success:
        break

    # Balik frame secara horizontal (flip)
    frame = cv2.flip(frame, 1)

    # Deteksi tangan dan wajah
    frame, landmarkList = detector.findHandLandMarks(frame, draw=True)
    frame, faceList = detector.findFaces(frame, draw=True)

    # Tampilkan frame
    cv2.imshow("Hand and Face Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) == ord('q'):
        break

# Lepaskan objek webcam dan tutup semua jendela
webcam.release()
cv2.destroyAllWindows()