# Face Emotion Counter 

Real-time face detection + deep learning emotion recognition using your webcam.

## What it does
- Detects all faces in the webcam frame using OpenCV Haar Cascades
- Predicts the **emotion** of each face (happy / sad / angry / surprise / fear / disgust / neutral) using a CNN via DeepFace
- Colour-codes each face box by emotion
- Tracks max faces seen and a full emotion log for the session
- Prints a session summary with emotion breakdown when you quit

## Setup

```bash
pip install opencv-python deepface tf-keras
```

## Run

```bash
python face_emotion_counter.py
```

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit and show session summary |
| `S` | Save screenshot to `screenshots/` folder |
| `R` | Reset session counter and emotion log |

## How it works

```
Webcam frame
    │
    ▼
Grayscale conversion
    │
    ▼
Haar Cascade face detection  ←── OpenCV (fast, CPU)
    │
    ▼
For each face ROI (every 5 frames):
    │
    ▼
DeepFace CNN analysis  ←── Pre-trained FER model (deep learning)
    │
    ▼
dominant_emotion label
    │
    ▼
Draw coloured box + label on frame
```

## Files

```
face_emotion_counter/
├── face_emotion_counter.py   ← main script (only file you need)
├── screenshots/              ← auto-created when you press S
└── README.md
```

## Emotions detected

| Emotion  | Box colour  |
|----------|-------------|
| Happy    | 🟢 Green   |
| Sad      | 🟤 Brown   |
| Angry    | 🔴 Red     |
| Surprise | 🔵 Cyan    |
| Fear     | 🟣 Purple  |
| Disgust  | 🟢 Dark green |
| Neutral  | ⚪ Gray    |


**Key ML concepts this project demonstrates:**

1. **Computer Vision pipeline** — frame capture → preprocessing → detection → classification
2. **Transfer learning** — DeepFace uses a CNN pre-trained on FER2013 dataset (35,000+ labelled facial images)
3. **Real-time inference** — model runs every N frames with a sliding cache to keep FPS smooth
4. **Feature extraction** — Haar Cascades extract face ROIs as features for the deep model
5. **Multi-class classification** — 7 emotion classes, softmax output layer
