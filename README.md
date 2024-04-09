# facial-recognition

A simple Python facial recognition app built using OpenCV

This script performs facial recognition using the `face_recognition` library. It can identify faces in an image loaded from a file or captured in real-time using a webcam. Identified faces can be saved to expand the known faces database, with the option for user feedback to improve recognition accuracy.

## Features

- Load images from a file or capture in real-time with a webcam.
- Identify faces using either the Histogram of Oriented Gradients (HOG) or Convolutional Neural Network (CNN) model.
- Interactive feedback loop for learning and saving new faces.
- Save identified faces with the option to include a margin for better detection on reload.

## Prerequisites

### Libraries

Before running the script, you need to install the following:

- Python 3.x
- `opencv-python` OpenCV library
- `face_recognition` library
- `face_recognition_models` (for face recognition models used by the face_recognition library)
- `numpy` (used by OpenCV and in the script)
- `matplotlib` (for displaying images)
- `python-dotenv` (for loading environment variables)

You can install the required libraries using `pip`:

```bash
pip install python-dotenv opencv-python numpy matplotlib face_recognition git+https://github.com/ageitgey/face_recognition_models
```

Note: you may need to install `cmake` and `python setuptools` to complete the installation of the face recognition models.

### Environment variables

Before running the script, rename `.env.example` to `.env` and update the variables to your desired settings.

## Usage

To use the script, follow these steps:

1. Clone the repository or download the script to your local machine.
2. Rename `.env.example` to `.env`
2. Run the script from the terminal:

```bash
python facial_recognition.py
```

4. Follow the interactive prompts to choose between using a webcam or loading an image from a file.
5. If an identified face is correct, confirm the match; otherwise, provide the correct name or mark it as unknown.
6. Identified faces will be saved and added to the known faces database for future recognition.

## Configuration

You can adjust the following settings in the script according to your needs:

- `model`: Choose between "default" (HOG) or "cnn" for the face detection model.
- `tolerance`: Set the strictness of face comparison (lower is stricter).
- `match_threshold`: Define the threshold for considering a match (distance-based).

## Known Issues

n/a

## License

Apache 2.0

## Contributions

Contributions are welcome. Please open an issue or submit a pull request with your suggested changes.