import face_recognition
import cv2
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
model = os.getenv('MODEL', 'default')  # "default" or "cnn" (cnn requires more GPU)
tolerance = float(os.getenv('TOLERANCE', '0.6'))  # Lower values make the recognition more strict, default is 0.6
match_threshold = float(os.getenv('MATCH_THRESHOLD', '0.5'))  # Threshold to consider a match, default is 0.5
feedback_loop = os.getenv('FEEDBACK_LOOP', 'True') == 'True'  # Enable or disable the feedback loop to learn and save new faces to expand the known faces database
face_database_dir = os.getenv('FACE_DATABASE_DIR', 'face_database')  # Where to store the known faces

def load_known_faces(known_faces_dir):
    """
    Loads the known faces from the specified directory and returns the encodings and names.
    """
    known_face_encodings = []
    known_face_names = []
    os.makedirs(known_faces_dir, exist_ok=True)
    for person_dir in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, person_dir)
        if os.path.isdir(person_path):
            print(f"Loading faces for {person_dir}")
            for filename in os.listdir(person_path):
                print(f"Loading {filename}")
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    face_image = face_recognition.load_image_file(os.path.join(person_path, filename))
                    face_encodings = face_recognition.face_encodings(face_image, model=model)
                    if len(face_encodings) == 0:
                        print(f"No face found in {os.path.join(person_path, filename)}")
                        continue
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_dir)
    return known_face_encodings, known_face_names

def save_face_image(img, face_location, name, margin=20):
    """
    Saves the cropped face image to the specified directory (either known or unknown).
    Includes a margin around the cropped face to ensure better face detection on reload.
    """
    top, right, bottom, left = face_location
    top = max(0, top - margin)
    right = min(img.shape[1], right + margin)
    bottom = min(img.shape[0], bottom + margin)
    left = max(0, left - margin)

    face_image = img[top:bottom, left:right]

    path = os.path.join(face_database_dir, "." if name == "unknown" else "known", name)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}_{np.random.randint(0, 1e6)}.jpg")

    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, face_image_bgr)
    print(f"Saved new face to {file_path}")

def display_face_image(img, face_location):
    """
    Displays a cropped face image using OpenCV's imshow. 
    This method is more compatible with standard Python environments.
    """
    top, right, bottom, left = face_location
    face_image = img[top:bottom, left:right]
    cv2.imshow('Facial Recognition', cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_face_image_with_matplotlib(img, face_location):
    """
    Displays a cropped face image using Matplotlib's imshow.
    This method is more compatible with Jupyter Notebooks and IPython environments.
    """
    top, right, bottom, left = face_location
    face_image = img[top:bottom, left:right]
    
    # Convert from BGR (OpenCV default) to RGB (Matplotlib default)
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(5, 5))  # You can adjust the figure size as needed
    plt.imshow(face_image)
    plt.axis('off')  # Hide the axis
    plt.show()

def distance_to_confidence(distance, max_distance=1.0):
    """
    Converts the distance between face encodings to a confidence percentage.
    """
    distance = min(distance, max_distance)
    confidence = 1.0 - (distance / max_distance)
    confidence_percentage = round(confidence * 100)
    return confidence_percentage

def recognize_faces_in_image(known_face_encodings, known_face_names, img):
    """
    Recognizes the faces in the image and displays the result. If the feedback loop is enabled, the user can provide
    input to correct the automatic identification. If the face is not recognized, the user can provide a new name.
    """
    img_face_locations = face_recognition.face_locations(img, model=model)
    img_face_encodings = face_recognition.face_encodings(img, known_face_locations=img_face_locations, model=model)

    if len(img_face_encodings) == 0:
        print("No faces found in the image.")
        return known_face_encodings, known_face_names

    for index, (face_encoding, face_location) in enumerate(zip(img_face_encodings, img_face_locations)):
        print(f"Processing face {index + 1}/{len(img_face_encodings)} at location {face_location}")

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
        else:
            best_match_index = -1
            print("No known faces to compare with.")

        display_face_image_with_matplotlib(img, face_location)

        identified_name = "unknown"  # Default to unknown
        if best_match_index > 0 and matches[best_match_index]:
            identified_name = known_face_names[best_match_index]
            best_match_distance = face_distances[best_match_index]
            print(f"best match distance {best_match_distance}")
            confidence = distance_to_confidence(best_match_distance)
            if best_match_distance < match_threshold:
                print(f"Identified as {identified_name} with confidence {confidence}%")
            else:
                print(f"Face detected, but no good match found. Best match was {identified_name} with confidence {confidence}%")
                identified_name = "unknown"
        else:
            print("No known match found")

        if feedback_loop:
            name = None
            if identified_name != "unknown":
                correct = input("Is this {identified_name}? (y/n): ")
                if correct.lower() == 'n':
                    save = input("Save? (y/n): ")
                    if save.lower() == 'y':
                        selected_name = input("Enter the correct name, or leave blank if unknown: ")
                        if selected_name.strip() != "":
                            name = selected_name
                            # Add the new face encoding and name to the known lists
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(name)
                        else:
                            name = "unknown"
                elif correct.lower() == 'y':
                    save = input("Save? (y/n): ")
                    if save.lower() == 'y':
                        name = identified_name
                        # Add the new face encoding and name to the known lists
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(name)                        
            else:
                save = input(f"Save? (y/n): ")
                if save.lower() == 'y':
                    selected_name = input("Enter the name, or leave blank if unknown: ")
                    if selected_name.strip() != "":
                        name = selected_name
                        # Add the new face encoding and name to the known lists
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(name)
                    else:
                        name = "unknown"

            if feedback_loop and name != None:
                # Save face image to the filesystem under the final
                print(f"Saving face image for {name}")
                save_face_image(img, face_location, name)

    return known_face_encodings, known_face_names

def capture_image_from_webcam():
    """
    Captures an image from the webcam and returns it.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    time.sleep(1)  # Warm-up camera
    ret, img = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read frame from webcam.")
        return None

    return img

def main():
    known_face_encodings, known_face_names = load_known_faces(face_database_dir+"/known")

    while True:
        choice = input("Would you like to use the webcam (w) or specify a file (f) or quit (q) ? ")

        if choice.lower() == 'w':
            img = capture_image_from_webcam()
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif choice.lower() == 'f':
            filename = input("Enter the path to the image file: ")
            try:
                img = face_recognition.load_image_file(filename.strip())
            except Exception as e:
                print(f"Could not load image: {e}")
                continue
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid option selected.")
            continue

        recognize_faces_in_image(known_face_encodings, known_face_names, img)

if __name__ == "__main__":
    main()
