# facial-recognition watch

## Description

Facial Recognition Watch REST API built with FastAPI

It uses the face_recognition library for the core functionality of processing and recognizing faces in images. The API can accept images through HTTP requests, identify faces in the received images, and compare them to a database of known faces.

The API is designed to be flexible and scalable, capable of handling real-time facial recognition tasks in various applications.

## Key Features

1. **Facial Recognition**: The script uses the `face_recognition` library to perform facial recognition tasks. This includes identifying faces in images and comparing them to a database of known faces.

2. **Configuration Flexibility**: The script allows for configuration of the facial recognition model, tolerance, and face database directory through environment variables. This provides flexibility in tuning the system's performance and accuracy.

3. **Known Faces Database**: The script maintains a database of known faces, stored as encodings, names, and filenames. These are loaded from a specified directory when the app starts.

4. **REST API**: The script is designed to work with the FastAPI framework, which allows it to serve as a web API. This enables other applications to interact with the facial recognition system over HTTP.

5. **File Handling**: The script uses the `os` and `cv2` libraries for file and image handling. This includes creating directories, reading files, and processing images.

## Dependencies

This project uses several Python libraries:

1. **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. It's used to create the REST API for the facial recognition system.

2. **Uvicorn**: An ASGI server that runs your FastAPI application. It's needed to serve the FastAPI application and handle incoming HTTP requests.

3. **OpenCV (opencv-python-headless)**: A library used for real-time computer vision. This headless variant doesn't include any GUI functionality, which is not needed for this API. It's used to process images for the facial recognition system.

4. **face_recognition**: A library for performing facial recognition tasks, including face detection and face recognition. It's the core library used for the facial recognition system in this API.

5. **python-dotenv**: This library allows the application to read from a `.env` file where environment variables can be stored. It's used to manage environment variables for the facial recognition model, tolerance, and face database directory.

## Installation

Follow these steps to install and set up the project:

1. **Clone the Repository**: First, clone the repository to your local machine using git. Open your terminal and run the following command:

    ```bash
    git clone https://github.com/timler/facial-recognition.git
    ```

2. **Navigate to the Project Directory**: Change your current directory to the project directory:

    ```bash
    cd facial-recognition
    ```

3. **Create a Virtual Environment (Optional)**: It's recommended to create a virtual environment to keep the project's dependencies isolated from your system. You can do this with the following commands:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4. **Install the package**: Install the project package. This will also install the dependencies specified in `setup.py`:

    ```bash
    pip3 install .
    ```

    If you want to make changes to the project and test it locally, you can install it in editable mode:

    ```bash
    pip3 install -e .
    ```
    
    To install `face_recognition_models` you will need to have CMake installed, which is required to build the `dlib` package which is a dependency of `face_recognition_models`.

    On Linux, run the following commands:
    ```bash
    sudo apt-get update
    sudo apt-get install cmake
    sudo apt-get install build-essential
    sudo apt-get install python3-dev
    pip3 install wheel
    ```

    On MacOs, run these commands:
    ```bash
    brew update
    brew install cmake
    pip3 install wheel
    ```

5. **Set Environment Variables**: Set the necessary environment variables for the facial recognition model, tolerance, and face database directory. You can do this in your terminal or by creating a `.env` file in the project directory. 

    To create the `.env` file, you can copy the `.env.example` file and rename it to `.env`. Then, update the values in the `.env` file with your desired configuration.

    ```bash
    cp .env.example .env
    ```

6. **Run the Application**: Finally, you can run the application with the following command:

    ```bash
    uvicorn watch.api:app --reload
    ```

    The `--reload` flag enables hot reloading, which means the server will automatically update whenever you make changes to the code.

Now, the facial recognition API should be running at [http://localhost:8000](http://localhost:8000).

## API Endpoints

This application has the following API endpoints:

1. **`POST /identify_faces`**: This endpoint accepts a Base64 encoded image and identifies the faces in the image using known face encodings and names. It returns the identified faces with the name, image, matching image, and confidence in the match expressed as a percentage.

2. **`POST /save_face`**: This endpoint accepts a Base64 encoded image and a name, and saves the face from the image to the database with the given name.

3. **`GET /get_images`**: This endpoint returns a list of all saved face images for a person given a name. If no name or name is "unknown", then it returns all the unknown face images.

4. **`POST /delete_face`**: This endpoint accepts a face image URL and deletes the corresponding face image from the database.

5. **`POST /label_face`**: This endpoint accepts a face image URL and a name, and labels the face in the image with the given name.

For more details about the API endpoints and to test them, please run the application and visit [http://localhost:8000/docs](http://localhost:8000/docs) in your web browser. This will open the automatically generated interactive API documentation (Swagger UI) where you can try out the endpoints directly.

## Development

### Testing

To run the tests install the dependencies:

```bash
pip3 install pytest requests httpx
```

Create an .env file from .env.example

```bash
cp .env.example tests/.env
```

Then run pytest:

```bash
pytest tests/test_api.py -s
```

## Known Issues

n/a

## License

Apache 2.0

## Contributions

Contributions are welcome. Please open an issue or submit a pull request with your suggested changes.

## Contact

Visit my website [dagmartimler.com](https://www.dagmartimler.com) for more information.
