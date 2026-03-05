import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "traffic_hand_signal_cnn.h5"
DATASET_PATH = "DataSet"
IMG_SIZE = (224, 224)

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# ===============================
# CLASS NAMES (MATCH TRAINING)
# ===============================
class_names = sorted(os.listdir(DATASET_PATH))
print("Classes:", class_names)

# ===============================
# PREDICTION SMOOTHING
# ===============================
prediction_queue = deque(maxlen=5)

# ===============================
# START WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam started | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural view
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # ===============================
    # ROI (LARGER CENTER REGION)
    # ===============================
    x1, y1 = int(w * 0.1), int(h * 0.1)
    x2, y2 = int(w * 0.9), int(h * 0.9)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = frame[y1:y2, x1:x2]


    # ===============================
    # PREPROCESS
    # ===============================
    img = cv2.resize(roi, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ===============================
    # PREDICT
    # ===============================
    preds = model.predict(img, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]

    prediction_queue.append(class_id)

    # Majority vote (stability)
    final_class = max(set(prediction_queue), key=prediction_queue.count)
    label = class_names[final_class]
    conf_text = f"{confidence*100:.2f}%"

    # ===============================
    # DISPLAY
    # ===============================
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"{label} ({conf_text})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Show ONE signal inside green box",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Traffic & Hand Signal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam closed")
