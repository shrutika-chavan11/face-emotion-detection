import cv2
import time
import os
from collections import Counter
from deepface import DeepFace

SCALE_FACTOR   = 1.1
MIN_NEIGHBORS  = 8
MIN_FACE_SIZE  = (80, 80)
ANALYZE_EVERY  = 5          # run DeepFace every N frames (keeps it smooth)
SCREENSHOT_DIR = "screenshots"


EMOTION_COLORS = {
    "happy":    (0,   220,  80),
    "sad":      (200,  80,   0),
    "angry":    (0,    0,  220),
    "surprise": (0,   200, 220),
    "fear":     (180,  0,  180),
    "disgust":  (0,   160,  40),
    "neutral":  (180, 180, 180),
}
DEFAULT_COLOR = (100, 100, 100)

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)


max_faces      = 0
frame_count    = 0
screenshot_n   = 0
session_start  = time.time()
emotion_log    = []           # list of (timestamp, emotion) for every face seen
face_emotions  = {}           # face_index -> last detected emotion string


def analyze_emotions(frame, faces):
    """
    Run DeepFace emotion analysis on each detected face region.
    Returns a dict  {face_index: emotion_label}.
    """
    results = {}
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = frame[y:y + h, x:x + w]
        try:
            analysis = DeepFace.analyze(
                face_roi,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            # DeepFace returns a list when multiple faces found inside the ROI
            if isinstance(analysis, list):
                analysis = analysis[0]
            emotion = analysis["dominant_emotion"]
        except Exception:
            emotion = "neutral"
        results[i] = emotion
    return results


def draw_overlay(frame, faces, emotions, max_f, elapsed):
    """Draw face boxes, emotion labels, and HUD onto frame (in-place)."""
    face_count = len(faces)

    for i, (x, y, w, h) in enumerate(faces):
        emotion = emotions.get(i, "neutral")
        color   = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background pill
        label     = f"Face {i+1}: {emotion}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        pad = 4
        cv2.rectangle(
            frame,
            (x, y - th - 2*pad),
            (x + tw + 2*pad, y),
            color, -1
        )
        cv2.putText(
            frame, label,
            (x + pad, y - pad),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )


    hud_lines = [
        f"Faces now : {face_count}",
        f"Max faces : {max_f}",
        f"Time      : {int(elapsed)}s",
        "",
        "S = screenshot",
        "R = reset  |  Q = quit",
    ]
    for row, text in enumerate(hud_lines):
        cv2.putText(
            frame, text,
            (14, 36 + row * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, text,
            (14, 36 + row * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 1, cv2.LINE_AA
        )


print("Face Emotion Counter running — press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    frame_count += 1
    elapsed = time.time() - session_start

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE,
    )

    # Update emotion predictions every ANALYZE_EVERY frames
    if frame_count % ANALYZE_EVERY == 0 and len(faces) > 0:
        face_emotions = analyze_emotions(frame, faces)
        # Log emotions for session summary
        for emotion in face_emotions.values():
            emotion_log.append((elapsed, emotion))

    max_faces = max(max_faces, len(faces))

    draw_overlay(frame, faces, face_emotions, max_faces, elapsed)

    cv2.imshow("Face Emotion Counter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        screenshot_n += 1
        path = os.path.join(SCREENSHOT_DIR, f"capture_{screenshot_n:03d}.jpg")
        cv2.imwrite(path, frame)
        print(f"Screenshot saved → {path}")

    elif key == ord('r'):
        max_faces = 0
        face_emotions = {}
        emotion_log.clear()
        session_start = time.time()
        print("Session reset.")


cap.release()
cv2.destroyAllWindows()

total_time = time.time() - session_start
emotion_counts = Counter(e for _, e in emotion_log)


print("        SESSION SUMMARY")
print(f"  Duration        : {int(total_time)}s")
print(f"  Max faces seen  : {max_faces}")
print(f"  Screenshots     : {screenshot_n}")
print(f"  Emotions logged : {sum(emotion_counts.values())}")
print()
if emotion_counts:
    print("  Emotion breakdown:")
    for emotion, count in emotion_counts.most_common():
        bar = "█" * min(count, 30)
        print(f"    {emotion:<10} {bar} {count}")
