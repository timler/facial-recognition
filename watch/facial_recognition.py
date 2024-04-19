import os
from io import BytesIO
import face_recognition
import logging
import cv2
import random

import watch.image_converter as image_converter

# Initialize the logger
logger = logging.getLogger(__name__)

class FacialRecognition:
    def __init__(self, model='default', tolerance=0.6, face_database_dir='face_database'):
        """
        Initializes the FacialRecognition object with the specified parameters.

        Args:
            model (str): "default" or "cnn" (cnn requires more GPU)
            tolerance (float): The tolerance for face recognition. Defaults to 0.6.  Lower values make the recognition more strict.
            face_database_dir (str): The directory where the face database is stored. Defaults to 'face_database'.
        """
        self.model = model
        self.tolerance = tolerance
        self.face_database_dir = face_database_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_filenames = []

        # Initialise the face database
        self._load_known_faces()

    def _load_known_faces(self):
        """
        Loads the known faces from the specified directory and returns the encodings, names and filenames.
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_filenames = []
        known_faces_dir = os.path.join(self.face_database_dir, "known")
        logger.info(f"Loading known faces in {known_faces_dir}...")
        os.makedirs(known_faces_dir, exist_ok=True)
        for person_dir in os.listdir(known_faces_dir):
            person_path = os.path.join(known_faces_dir, person_dir)
            if os.path.isdir(person_path):
                logger.debug(f"Loading faces for {person_dir}")
                for filename in os.listdir(person_path):
                    filename = filename
                    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".webp"):
                        logger.debug(f"Loading {filename}")
                        face_image = face_recognition.load_image_file(os.path.join(person_path, filename))
                        face_encodings = face_recognition.face_encodings(face_image, model=self.model)
                        if len(face_encodings) == 0:
                            logger.info(f"No face found in {os.path.join(person_path, filename)}")
                            continue
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(person_dir.title())
                        self.known_face_filenames.append(filename)

    def _distance_to_confidence(self, distance, max_distance=1.0):
        """
        Converts the distance between face encodings to a confidence percentage.
        """
        distance = min(distance, max_distance)
        confidence = 1.0 - (distance / max_distance)
        confidence_percentage = round(confidence * 100)
        return confidence_percentage

    def _crop_image_to_face(self, img, face_location, margin=20):
        """
        Crops the image to the specified face location with an optional margin.
        """
        top, right, bottom, left = face_location
        top = max(0, top - margin)
        right = min(img.shape[1], right + margin)
        bottom = min(img.shape[0], bottom + margin)
        left = max(0, left - margin)
        return img[top:bottom, left:right]

    def recognize_faces_in_image(self, image_base64):
        """
        Identifies the faces in the image using the known face encodings and names.

        Args:
            image_base64 (str): The Base64 encoded image.

        Returns:
            list: A list of dictionaries representing the identified faces. Each dictionary contains the following keys:
                - "name" (str): The name of the identified face.
                - "image" (str): The Base64 encoded image of the identified face.
                - "matching_image" (str): The Base64 encoded image of the matching face from the known face database.
                - "confidence" (float): The confidence percentage of the match.

        Raises:
            Exception: If no face is found in the image.
        """
        identified_faces = []

        # Convert the Base64 encoded image to an image
        image_format = image_converter.base64_image_format(image_base64)
        image_bytes = image_converter.base64_to_image_buffer(image_base64)
        image_array = face_recognition.load_image_file(image_bytes)

        # Find all the faces in the image and compute their encodings
        img_face_locations = face_recognition.face_locations(image_array, model=self.model)
        img_face_encodings = face_recognition.face_encodings(image_array, known_face_locations=img_face_locations, model=self.model)
        if len(img_face_locations) == 0:
            raise Exception("No face found in the image")

        for index, (face_encoding, face_location) in enumerate(zip(img_face_encodings, img_face_locations)):
            logger.debug(f"Processing face {index + 1}/{len(img_face_encodings)} at location {face_location}")

            # Compare the face encoding with the known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            # Crop the image to the face location and define the result
            face_image_array = self._crop_image_to_face(image_array, face_location)
            identified_face = {
                "name": "unknown",
                "image": image_converter.image_array_to_base64(face_image_array, image_format),
                "matching_image": "",
                "confidence": 0
            }

            # Determine the best match and confidence
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
            else:
                best_match_index = -1
                logger.info("No known faces to compare with")

            # If a match is found, set the identified face details
            if best_match_index >= 0 and matches[best_match_index]:
                identified_name = self.known_face_names[best_match_index]
                matched_face_filename = os.path.join(self.face_database_dir, "known", identified_name.lower(), self.known_face_filenames[best_match_index])
                confidence = self._distance_to_confidence(face_distances[best_match_index])
                identified_face["name"] = identified_name
                identified_face["matching_image"] = image_converter.image_file_to_base64(matched_face_filename)
                identified_face["confidence"] = confidence
                logger.info(f"Identified as {identified_name} with confidence {confidence}%")
            else:
                logger.info("No known match found")
            
            identified_faces.append(identified_face)

        return identified_faces

    def save_face_image(self, image_base64, name):
        """
        Saves the specified image to the face database directory with the specified name.

        Parameters:
        - image_base64 (str): The Base64 encoded image.
        - name (str): The name to be associated with the image.

        Returns:
        - str: The file path of the saved image.

        Raises:
        - Exception: If no face is found in the image.
        """
        # Convert the Base64 encoded image to an image
        image_bytes = image_converter.base64_to_image_buffer(image_base64)
        image_array = face_recognition.load_image_file(image_bytes)
        
        # Identify the face in the image
        face_encodings = face_recognition.face_encodings(image_array, model=self.model)
        if len(face_encodings) == 0:
            raise Exception("No face found in the image")
        
        # Crop the image to the face location
        img_face_locations = face_recognition.face_locations(image_array, model=self.model)
        if len(img_face_locations) == 0:
            raise Exception("No face found in the image")
        face_image_array = self._crop_image_to_face(image_array, img_face_locations[0])
        
        # Save the image to the specified directory
        name = name.lower().strip()
        filename = f"{name}_{random.randint(0, 1000000)}.jpg"
        sub_folder = "unknown" if name == "unknown" else os.path.join("known", name)
        full_path = os.path.join(self.face_database_dir, sub_folder)
        os.makedirs(full_path, exist_ok=True)
        file_path = os.path.join(full_path, filename)
        face_image_bgr = cv2.cvtColor(face_image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, face_image_bgr)
        logger.info(f"Saved new face to {file_path}")

        # Add the new face to the known faces
        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name.title())
        self.known_face_filenames.append(filename)

        return os.path.join(sub_folder, filename)

    def label_image(self, face_image_url, name):
        """
        Labels the specified image with the specified name.

        Args:
            face_image_url (str): The URL of the face image to be labeled.
            name (str): The name to be associated with the labeled image.

        Returns:
            str: The path of the labeled image relative to the face database directory.

        Raises:
            Exception: If the specified image is not found in the face database directory.
        """
        if os.path.exists(os.path.join(self.face_database_dir, face_image_url)):
            # Determine the sub-folder to save the image
            if name is None or name == "" or name.strip() == "unknown":
                sub_folder = "unknown"
            else:
                name = name.lower().strip()
                sub_folder = os.path.join("known", name)
            path = os.path.join(self.face_database_dir, sub_folder)
            os.makedirs(path, exist_ok=True)

            # Move the image to the specified directory
            old_file_path = os.path.join(self.face_database_dir, face_image_url)
            new_filename = f"{name}_{random.randint(0, 1000000)}.jpg"
            new_file_path = os.path.join(path, new_filename)
            os.rename(old_file_path, new_file_path)
            logger.info(f"Labeled image {old_file_path} as {new_file_path}")

            # Reload the known faces
            self._load_known_faces()

            return os.path.join(sub_folder, new_filename)
        else:
            raise Exception(f"Image {os.path.join(self.face_database_dir, face_image_url)} not found.")

    def _get_name_from_filename(self, filename):
        name = filename.split(".")[0].split("_")[0]
        if name != "unknown":
            name = name.title()
        return name

    def get_all_images(self, name):
        """
        Retrieves all the images known for the specified name, or all unknown images if no name specified or "unknown".

        Args:
            name (str): The name of the person whose images are to be retrieved. If None or "unknown", retrieves all unknown images.

        Returns:
            list: A list of dictionaries containing information about each image. Each dictionary has the following keys:
                - "face_image_url" (str): The URL of the face image.
                - "name" (str): The name of the person in the image.
                - "image_base64" (str): The base64-encoded image data.

        """
        # Determine the sub-folder to search for images
        if name is None or name == "" or name.strip() == "unknown":
            sub_folder = "unknown"
        else:
            name = name.lower().strip()
            sub_folder = os.path.join("known", name)
        path = os.path.join(self.face_database_dir, sub_folder)

        # Retrieve all the images in the specified directory
        images = []
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".webp"):
                    with open(os.path.join(path, filename), "rb") as f:
                        images.append({
                            "face_image_url": os.path.join(sub_folder, filename),
                            "name": self._get_name_from_filename(filename),
                            "image_base64": image_converter.image_file_to_base64(f)
                        })
        logger.info(f"Retrieved {len(images)} images for name: {name}")

        return images

    def delete_image(self, filename):
        """
        Deletes the specified image file.

        Parameters:
        - filename: A string representing the name of the image file to be deleted.

        Returns:
        - A dictionary containing the details of the deleted image:
            - face_image_url: The filename of the deleted image.
            - name: The name extracted from the filename.
            - image_base64: The base64 encoded representation of the deleted image.

        Raises:
        - Exception: If the image file specified by the filename does not exist.
        """
        file_path = os.path.join(self.face_database_dir, filename)
        if os.path.exists(file_path):
            name = self._get_name_from_filename(os.path.basename(file_path))
            image_base64 = image_converter.image_file_to_base64(file_path)
            os.remove(file_path)
            logger.info(f"Deleted image {file_path}")
            self._load_known_faces() # Reload the known faces
            return {
                "face_image_url": filename,
                "name": name,
                "image_base64": image_base64
            }
        else:
            raise Exception(f"Image {filename} not found.")
