import face_recognition
import logging

import watch.image_converter as image_converter
from watch.face_database import FaceDatabase

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
        self.face_database = FaceDatabase(face_database_dir)
        self._encode_known_faces()

    def _encode_known_faces(self):
        """
        Encodes the known faces in the face database.
        """
        self.known_face_encodings = []
        for index, filename in enumerate(self.face_database.known_face_file_urls):
            file = self.face_database.get_actual_file_path_from_url(filename)
            face_image = face_recognition.load_image_file(file)
            face_encodings = face_recognition.face_encodings(face_image, model=self.model)
            if len(face_encodings) == 0:
                logger.info(f"No face found in {file}")
                self.face_database.remove_face_at_index(index)
            else:
                self.known_face_encodings.append(face_encodings[0])

    def _distance_to_confidence(self, distance, max_distance=1.0):
        """
        Converts the distance between face encodings to a confidence percentage.
        """
        distance = min(distance, max_distance)
        confidence = 1.0 - (distance / max_distance)
        confidence_percentage = round(confidence * 100)
        return confidence_percentage

    def _crop_image_to_face(self, img, face_location, margin=100):
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
                identified_name = self.face_database.get_face_name(best_match_index)
                idenfitied_file_url = self.face_database.get_face_file_url(best_match_index)
                matched_face_filename = self.face_database.get_actual_file_path_from_url(idenfitied_file_url)
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
        img_face_locations = face_recognition.face_locations(image_array, model=self.model)
        if len(img_face_locations) == 0:
            raise Exception("No face found in the image")
        if len(img_face_locations) > 1:
            raise Exception("Multiple faces found in the image. Please use cropped images with only one face.")
        
        # Crop the image to the face location
        face_image_array = self._crop_image_to_face(image_array, img_face_locations[0])
        
        # Save the face image to the face database
        file_path = self.face_database.save_face_image(name, face_image_array)

        # Add the new face to the known faces
        self.known_face_encodings.append(face_encodings[0])

        return file_path

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
        new_url = self.face_database.label_image(face_image_url, name)
        self._encode_known_faces() # Re-encode the known faces
        return new_url

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
        images = []
        image_file_urls = self.face_database.get_all_images(name)
        for url in image_file_urls:
            images.append({
                "face_image_url": url,
                "name": self.face_database.get_name_from_filename(url),
                "image_base64": image_converter.image_file_to_base64(self.face_database.get_actual_file_path_from_url(url))
            })

        logger.info(f"Retrieved {len(images)} images for name: {name}")
        return images

    def delete_image(self, face_image_url):
        """
        Deletes the specified image file.

        Parameters:
        - face_image_url: The URL of the image file to be deleted.

        Returns:
        - A dictionary containing the details of the deleted image:
            - face_image_url: The filename of the deleted image.
            - name: The name extracted from the filename.
            - image_base64: The base64 encoded representation of the deleted image.

        Raises:
        - Exception: If the image file specified by the filename does not exist.
        """
        if self.face_database.file_exists(face_image_url):
            name = self.face_database.get_name_from_filename(face_image_url)
            actual_file_path = self.face_database.get_actual_file_path_from_url(face_image_url)
            image_base64 = image_converter.image_file_to_base64(actual_file_path)
            self.face_database.delete_image(face_image_url)
            self._encode_known_faces() # Re-encode the known faces
            return {
                "face_image_url": face_image_url,
                "name": name,
                "image_base64": image_base64
            }
        else:
            raise Exception(f"Image {face_image_url} not found.")
