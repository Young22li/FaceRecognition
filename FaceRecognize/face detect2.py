import cv2
import os
import time
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Picture folder path
folder_path = r"/Users/14350/PycharmProjects/PythonProject1/sample"

# Statistics
total_images = 0
detected_faces = 0
no_faces = 0

# Start Time
start_time = time.time()

# Create the Face Detection processor
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # Walk through all the images in the folder
    for filename in os.listdir(folder_path):
        if total_images >= 20000:
            break

        img_path = os.path.join(folder_path, filename)

        # Only.jpg,.png,.jpeg files are processed
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        total_images += 1
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Cannot open image {filename}. Skipping...")
            continue

        # Convert to RGB format (required by MediaPipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face dection
        results = face_detection.process(img_rgb)

        if results.detections:
            detected_faces += 1
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)

            # Showing result
            #cv2.imshow(f"Detected Faces - {filename}", img)
            #cv2.waitKey(500)  # 显示 500ms 后自动关闭
            #cv2.destroyAllWindows()
        else:
            no_faces += 1
            print(f"Can't Detect Face: {filename}")

# End time
end_time = time.time()
total_time = end_time - start_time

# Print Result
print(f"Total Images Processed: {total_images}")
print(f"Images with Detected Faces: {detected_faces}")
print(f"Images with No Faces: {no_faces}")
print(f"Total Processing Time: {total_time:.2f} seconds")