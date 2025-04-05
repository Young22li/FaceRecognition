import os
import cv2
import face_recognition
import numpy as np

# Training data directory (store photos of known people)
TRAIN_DIR = "/Users/14350/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/train"
TEST_DIR = "/Users/14350/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/test"

# Encode and file name for storing training data
known_encodings = []
known_names = []

count = 0
mis_match = 0
# Read the picture in the train directory and code it
for filename in os.listdir(TRAIN_DIR):
    image_path = os.path.join(TRAIN_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(filename)

# Iterate through the images in the test directory
for test_filename in os.listdir(TEST_DIR):
    test_image_path = os.path.join(TEST_DIR, test_filename)
    test_image = face_recognition.load_image_file(test_image_path)
    test_encodings = face_recognition.face_encodings(test_image)

    if not test_encodings:
        continue

    test_encoding = test_encodings[0]

    # Match
    matches = face_recognition.compare_faces(known_encodings, test_encoding, tolerance=0.45)
    face_distances = face_recognition.face_distance(known_encodings, test_encoding)
    best_match_index = np.argmin(face_distances)


    if matches[best_match_index]:
        count += 1
    else:
        print(f"No Successful: {test_filename} 与 {known_names[best_match_index]}")
        mis_match += 1

print(count)
print(mis_match)