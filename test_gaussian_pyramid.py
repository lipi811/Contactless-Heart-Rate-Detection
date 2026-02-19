import cv2
from src.face_detection import detect_face
from src.gaussian_pyramid import build_gaussian_pyramid

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = frame[y:y+h, x:x+w]

        pyramid = build_gaussian_pyramid(roi, levels=3)

        # Display pyramid levels
        for i, level in enumerate(pyramid):
            cv2.imshow(f"Gaussian Level {i}", level)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("STEP 3: Face ROI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
