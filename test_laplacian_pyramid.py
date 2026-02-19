import cv2
from src.face_detection import detect_face
from src.gaussian_pyramid import build_gaussian_pyramid
from src.laplacian_pyramid import build_laplacian_pyramid

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = frame[y:y+h, x:x+w]

        gp = build_gaussian_pyramid(roi, levels=3)
        lp = build_laplacian_pyramid(gp)

        for i, level in enumerate(lp):
            # Normalize for display
            display = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX)
            display = display.astype('uint8')
            cv2.imshow(f"Laplacian Level {i}", display)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("STEP 4: Face ROI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
