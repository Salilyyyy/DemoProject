import cv2
import os
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Directory containing reference images
reference_image_dir = 'pic1/'

# Function to compare two images
def compare_images(img1, img2):
    # Convert both images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Ensure both images have the same dimensions
    min_height = min(gray1.shape[0], gray2.shape[0])
    min_width = min(gray1.shape[1], gray2.shape[1])
    gray1 = gray1[:min_height, :min_width]
    gray2 = gray2[:min_height, :min_width]

    # Compute the Structural Similarity Index (SSI) using scikit-image
    ssim_index = compare_ssim(gray1, gray2)

    return ssim_index

# Load and preprocess reference images
reference_images = []
for filename in os.listdir(reference_image_dir):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(reference_image_dir, filename))
        reference_images.append(img)

# Initialize the video capture object (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]

        # Compare the cropped face with all reference images
        match_found = False
        for ref_image in reference_images:
            similarity = compare_images(face, ref_image)
            if similarity > 0.7:  # Adjust the threshold as needed
                match_found = True
                break

        if match_found:
            result_text = "Giang"
        else:
            result_text = "Unknown"

        # Draw a rectangle around the face and display the result
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detected faces and result
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()