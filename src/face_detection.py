import cv2

# Load Haar Cascade classifier for frontal face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(frame):
    """
    Detect faces in a video frame using Haar Cascade.

    Parameters:
        frame (numpy.ndarray): Input frame in BGR format

    Returns:
        faces (list): List of (x, y, w, h) face bounding boxes
    """

    # Convert to grayscale as Haar Cascade works on intensity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return faces
