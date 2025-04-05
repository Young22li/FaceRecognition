import os
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize MTCNN (Face detection) and FaceNet (Face recognition)
detector = MTCNN()
embedder = FaceNet()
count = 0
mis_match =0
# Training data directory (store photos of known people)
TRAIN_DIR = "/Users/14350/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/train"
TEST_DIR = "/Users/14350/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/test"

# Encode and file name for storing training data
known_encodings = []
known_names = []

# Read the picture in the train directory and code it
for filename in os.listdir(TRAIN_DIR):
    image_path = os.path.join(TRAIN_DIR, filename)
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb_image)
    if not faces:
        continue

    x, y, width, height = faces[0]['box']
    face = rgb_image[y:y + height, x:x + width]
    face = cv2.resize(face, (160, 160))
    encoding = embedder.embeddings(np.expand_dims(face, axis=0))[0]

    known_encodings.append(encoding)
    known_names.append(filename)

# Iterate through the images in the test directory
for test_filename in os.listdir(TEST_DIR):
    test_image_path = os.path.join(TEST_DIR, test_filename)
    test_image = cv2.imread(test_image_path)
    rgb_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb_test_image)
    if not faces:
        continue

    x, y, width, height = faces[0]['box']
    test_face = rgb_test_image[y:y + height, x:x + width]
    test_face = cv2.resize(test_face, (160, 160))
    test_encoding = embedder.embeddings(np.expand_dims(test_face, axis=0))[0]

    # Calculate the Euclidean distance to find the nearest match
    distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
    best_match_index = np.argmin(distances)
    min_distance = distances[best_match_index]


    # Threshold setting
    if min_distance < 0.8:
        #print(f"Success: {test_filename} 与 {known_names[best_match_index]} (距离: {min_distance:.2f})")
        count += 1

    else:
        print(f"No match found : {test_filename} and 与 {known_names[best_match_index]}")
        mis_match += 1


print(f"Success: {count}")
print(f"Unsuccess: {mis_match}")