import cv2
import face_recognition
import pickle
import numpy as np
import requests
import time
from datetime import datetime

ATTENDANCE_URL = "http://localhost:5000/attendance"

FRAME_SKIP = 3            # Process 1 of every 3 frames → fast
DOWNSCALE = 0.50          # 50% resolution for speed
ENCODING_EVERY = 2        # Encode only every 2nd detection
BOX_PERSISTENCE = 10      # Keep rectangles visible longer (stable display)

last_sent = {}
last_boxes = []
last_names = []
persist_counter = 0


# Attendance Sender
def send_attendance(name):
    now = time.time()
    if name in last_sent and now - last_sent[name] < 2:
        return

    payload = {
        "name": name,
        "timestamp": datetime.now().isoformat()
    }

    try:
        requests.post(ATTENDANCE_URL, json=payload, timeout=1)
        print(f"📨 Attendance sent → {payload}")
    except:
        print("❌ Attendance send failed")

    last_sent[name] = now


# ================================
# Main Face Recognition Loop
# ================================
def main():
    global last_boxes, last_names, persist_counter

    # Load encodings
    try:
        with open("encodings.pkl", "rb") as f:
            known_encodings = pickle.load(f)
        with open("names.pkl", "rb") as f:
            known_names = pickle.load(f)

        print("👍 Encodings loaded.")
    except:
        print("❌ ERROR: encodings.pkl or names.pkl missing.")
        return

    stream_url = "http://localhost:5000/stream"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ Cannot open stream")
        return

    print("🎥 Starting FAST recognition…")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Stream lost")
            break

        frame_count += 1
        display_frame = frame.copy()

        # -------------------------------------------------
        # 1. SKIP FRAMES → USE LAST KNOWN BOXES (NO LAG)
        # -------------------------------------------------
        if frame_count % FRAME_SKIP != 0:
            persist_counter -= 1

            # Draw previous boxes during skipped frames
            if persist_counter > 0:
                for (top, right, bottom, left), name in zip(last_boxes, last_names):
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, name, (left, top - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("FAST ESP32 Recognition", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # -------------------------------------------------
        # 2. DOWNSCALE → FAST DETECTION
        # -------------------------------------------------
        small = cv2.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detect faces
        locations = face_recognition.face_locations(rgb_small, model="hog")

        # Scale back up
        locations_full = []
        for (top, right, bottom, left) in locations:
            locations_full.append((
                int(top / DOWNSCALE),
                int(right / DOWNSCALE),
                int(bottom / DOWNSCALE),
                int(left / DOWNSCALE)
            ))

        # Reset temporary storage
        last_boxes = locations_full
        last_names = []

        # -------------------------------------------------
        # 3. Encode only in selected frames
        # -------------------------------------------------
        if frame_count % ENCODING_EVERY == 0:
            encodings = face_recognition.face_encodings(frame, known_face_locations=locations_full)

            for encoding in encodings:
                name = "Unknown"

                matches = face_recognition.compare_faces(known_encodings, encoding)
                if True in matches:
                    idx = matches.index(True)
                    name = known_names[idx]
                    send_attendance(name)

                last_names.append(name)

            # Persist longer to stabilize rectangle drawing
            persist_counter = BOX_PERSISTENCE

        # -------------------------------------------------
        # 4. Draw Boxes
        # -------------------------------------------------
        for (top, right, bottom, left), name in zip(last_boxes, last_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(display_frame, name, (left, top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("FAST ESP32 Recognition", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
