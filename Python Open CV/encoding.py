import os
import pickle
import cv2
import face_recognition
import numpy as np  # Essential for the fix

# --- Configuration ---
DATASET_DIR = "faces_dataset"
ENCODINGS_FILE = "encodings.pkl"
NAMES_FILE = "names.pkl"

known_encodings = []
known_names = []

print("\n🔍 Starting encoding process (Memory-Safe Mode)...\n")

# Loop through each person's folder
for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f" Processing: {person_name}")

    # Loop through each image
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)

        # Skip hidden/system files
        if img_file.startswith("."):
            continue

        try:
            # 1. Load with OpenCV
            bgr_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if bgr_image is None:
                print(f"    SKIP: Could not read {img_file}")
                continue

            # 2. Convert BGR to RGB
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # Force the data to be 8-bit integers (prevents float errors)
            rgb_image = rgb_image.astype('uint8')

            # Force the memory to be Contiguous (prevents "Unsupported Image Type" error)
            rgb_image = np.ascontiguousarray(rgb_image)

            # =========================================================

            # 3. Generate Encoding
            # We assume the image is already cropped to the face (from your capture script)
            # So we ask it to find the face within the 200x200 square
            boxes = face_recognition.face_locations(rgb_image, model="hog")

            # If no face found by HOG, use the whole image as the face location
            if not boxes:
                height, width, _ = rgb_image.shape
                boxes = [(0, width, height, 0)]

            encodings = face_recognition.face_encodings(rgb_image, known_face_locations=boxes)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"   ✔ Encoded {img_file}")
            else:
                print(f"   ✖ Failed to encode {img_file}")

        except Exception as e:
            print(f"   ⚠ Error processing {img_file}: {e}")

# Save the final data
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(known_encodings, f)

with open(NAMES_FILE, "wb") as f:
    pickle.dump(known_names, f)

print(f"\n Encoding complete.")
print(f"   → Total faces encoded: {len(known_encodings)}")

print(f"   → Saved to: {ENCODINGS_FILE}, {NAMES_FILE}")
