import os
import random
import logging

from PIL import Image

# Initialize the logger
logger = logging.getLogger(__name__)

class FaceDatabase:
    """
    Represents a face database that stores information about known faces in the file system.

    Attributes:
        face_database_dir (str): The directory path where the face database is stored.
        known_face_names (list): A list of names associated with the known faces.
        known_face_file_urls (list): A list of file URLs corresponding to the known faces.
    """
    def __init__(self, face_database_dir):
        self.face_database_dir = face_database_dir
        self.known_face_names = []
        self.known_face_file_urls = []
        self._load_known_faces()

    def _load_known_faces(self):
        """
        Loads the known faces from the specified directory and sets the names and filenames.
        """
        self.known_face_names = []
        self.known_face_file_urls = []
        known_faces_dir = os.path.join(self.face_database_dir, "known")
        logger.info(f"Loading known faces in {known_faces_dir}...")
        os.makedirs(known_faces_dir, exist_ok=True)
        for person_dir in os.listdir(known_faces_dir):
            person_path = os.path.join(known_faces_dir, person_dir)
            if os.path.isdir(person_path):
                logger.debug(f"Loading faces for {person_dir}")
                for filename in os.listdir(person_path):
                    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".webp"):
                        logger.debug(f"Loading {filename}")
                        self.known_face_names.append(person_dir.title())
                        self.known_face_file_urls.append(os.path.join("known", person_dir, filename))

    def _get_sub_folder(self, name):
        if name is None or name == "unknown" or name == "":
            return "unknown"
        else:
            name = name.lower().strip()
            return os.path.join("known", name)
        
    def _generate_file_name(self, name):
        name = name.lower().strip()
        return f"{name}_{random.randint(0, 1000000)}.jpg"
    
    def remove_face_at_index(self, index):
        """
        Removes a face from the database at the specified index. This is done
        when an image turns out not to have a recognizable face.

        Args:
            index (int): The index of the face to be removed.

        Raises:
            Exception: If the index is out of range.

        """
        if index < 0 or index >= len(self.known_face_names):
            raise Exception(f"Index {index} out of range.")
        del self.known_face_file_urls[index]
        del self.known_face_names[index]
    
    def get_actual_file_path_from_url(self, face_image_url):
        """
        Returns the actual file path given a URL of an image.

        Args:
            face_image_url (str): The URL of the face image.

        Returns:
            str: The actual file path of the face image.
        """
        return os.path.join(self.face_database_dir, face_image_url)
    
    def file_exists(self, face_image_url):
        """
        Checks if the file corresponding to the given face image URL exists on the filesystem.

        Args:
            face_image_url (str): The URL of the face image.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return os.path.exists(self.get_actual_file_path_from_url(face_image_url))
    
    def get_name_from_filename(self, filename):
        """
        Extracts the name of a person from a given filename.

        Args:
            filename (str): The filename from which to extract the name.

        Returns:
            str: The extracted name, can be "unknown" if the face is not labeled yet.

        """
        basename = os.path.basename(filename)
        name = basename.split(".")[0].split("_")[0]
        if name != "unknown":
            name = name.title()
        return name
    
    def get_face_file_url(self, index):
        """
        Returns the face image URL for a given index.

        Parameters:
        index (int): The index of the face file URL to retrieve.

        Returns:
        str: The face file URL at the specified index.
        """
        return self.known_face_file_urls[index]
    
    def get_face_name(self, index):
        """
        Returns the name associated with the face at the given index.

        Parameters:
        - index (int): The index of the face.

        Returns:
        - str: The name associated with the face.

        """
        return self.known_face_names[index]
    
    def save_face_image(self, name, image_array):
        """
        Saves a face image to the face database.

        Args:
            name (str): The name of the person associated with the face image.
            image_array (numpy.ndarray): The array representation of the face image.

        Returns:
            str: The file URL of the saved face image.
        """
        # Create the subfolder and file path
        name = name.lower().strip()
        filename = self._generate_file_name(name)
        sub_folder = self._get_sub_folder(name)
        full_path = os.path.join(self.face_database_dir, sub_folder)
        os.makedirs(full_path, exist_ok=True)
        file_path = os.path.join(full_path, filename)

        # Convert the image array to an image and save it
        image = Image.fromarray(image_array)
        image.save(file_path)        
        logger.info(f"Saved new face to {file_path}")

        # add the image to the known faces
        file_url = os.path.join(sub_folder, filename)
        self.known_face_names.append(name.title())
        self.known_face_file_urls.append(file_url)

        return file_url

    def label_image(self, face_image_url, name):
        """
        Moves a labeled face image to a new directory based on the provided name.

        Args:
            face_image_url (str): The URL of the face image to be labeled.
            name (str): The name associated with the face image.

        Returns:
            str: The url of the new labeled image.

        Raises:
            Exception: If the image specified by `face_image_url` is not found.
        """
        old_file_path = self.get_actual_file_path_from_url(face_image_url)
        if os.path.exists(old_file_path):
            # Get the new file path
            name = name.lower().strip()
            new_filename = self._generate_file_name(name)
            new_sub_folder = self._get_sub_folder(name)

            # Create the new folder if it does not exist
            new_path = os.path.join(self.face_database_dir, new_sub_folder)
            os.makedirs(new_path, exist_ok=True)

            # Move the image to the specified directory
            new_file_path = os.path.join(new_path, new_filename)
            os.rename(old_file_path, new_file_path)
            logger.info(f"Moved image {old_file_path} to {new_file_path}")

            # Reload the known faces
            self._load_known_faces()

            return os.path.join(new_sub_folder, new_filename)
        else:
            raise Exception(f"Image {old_file_path} not found.")
        
    def get_all_images(self, name=None):
        """
        Retrieve a list of all images in the face database.

        Args:
            name (str, optional): The name of the person to search for images. If not provided, all unknown images will be returned.

        Returns:
            list: A list of image urls relative to the face database directory.

        """
        images = []
        sub_folder = self._get_sub_folder(name)
        path = os.path.join(self.face_database_dir, sub_folder)
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".webp"):
                    images.append(os.path.join(sub_folder, filename))
        return images

    def delete_image(self, face_image_url):
        """
        Deletes the image file associated with the given face image URL.

        Args:
            face_image_url (str): The URL of the face image to be deleted.

        Raises:
            Exception: If the image file is not found.

        Returns:
            None
        """
        file_path = self.get_actual_file_path_from_url(face_image_url)
        if os.path.exists(file_path):
            # Delete the image
            os.remove(file_path)
            logger.info(f"Deleted image {file_path}")

            # Reload the known faces
            self._load_known_faces()
        else:
            raise Exception(f"Image {file_path} not found.")
