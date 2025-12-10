import cv2
import os
import requests
import numpy as np
import time  # Added for loop throttling

SNAPSHOT_URL = "http://localhost:5000/snapshot"
SAVE_DIR = "faces_dataset"
SAMPLES = 5
PADDING = 20  # Best Practice: Add space around the face


def ensure_folder(name):
    path = os.path.join(SAVE_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def main():
    name = input("Enter name: ").strip()
    if not name:
        print("❌ Name cannot be empty.")
        return

    save_path = ensure_folder(name)
    count = 0

    # Load detector once to save resources
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("\n📸 Starting CLEAN sample capture...\n")

    while count < SAMPLES:
        try:
            resp = requests.get(SNAPSHOT_URL, timeout=3)

            # 1. Check if server actually sent an image
            if resp.status_code != 200:
                print(f"⏳ Waiting for video stream... (Server: {resp.status_code})", end='\r')
                time.sleep(1)
                continue

            img_array = np.frombuffer(resp.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("⚠ Invalid image data received — skipping")
                continue

        except requests.exceptions.RequestException:
            print("⚠ Connection failed. Is the Node.js server running?")
            time.sleep(2)  # Prevent crash loop
            continue
        except Exception as e:
            print(f"⚠ Error: {e}")
            continue

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # 2. Sort to find largest face (Main Subject)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # 3. Best Practice: Add Padding (Margin)
        # This ensures chin/hair/ears are included for better recognition later
        h_img, w_img, _ = frame.shape
        x_pad = max(0, x - PADDING)
        y_pad = max(0, y - PADDING)
        w_pad = min(w_img - x_pad, w + (PADDING * 2))
        h_pad = min(h_img - y_pad, h + (PADDING * 2))

        # Crop with padding
        face = frame[y_pad: y_pad + h_pad, x_pad: x_pad + w_pad]

        # Resize for clean encoding
        try:
            face_resized = cv2.resize(face, (200, 200))

            filename = os.path.join(save_path, f"{name}_{count}.jpg")
            cv2.imwrite(filename, face_resized)

            count += 1
            print(f"✔ Saved {count}/{SAMPLES} | Location: {filename}")

            # Draw box on preview (green)
            cv2.rectangle(frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (0, 255, 0), 2)
            cv2.imshow("Capture", frame)

            # Wait 500ms so we don't capture 5 identical frames instantly
            # This forces the user to move slightly, creating a better dataset
            cv2.waitKey(500)

        except Exception as e:
            print(f"⚠ Resize error (face too close to edge?): {e}")

    print("\n✅ DONE — Clean samples captured.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()