import os
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from watch.facial_recognition import FacialRecognition
from fastapi import UploadFile

# Configuration
model = os.getenv('MODEL', 'default')  # "default" or "cnn" (cnn requires more GPU)
tolerance = float(os.getenv('TOLERANCE', '0.6'))  # Lower values make the recognition more strict, default is 0.6
face_database_dir = os.getenv('FACE_DATABASE_DIR', 'face_database')  # Where to store the known faces

# Create an instance of the FacialRecognition class
fr = FacialRecognition(model, tolerance, face_database_dir)

# Create the FastAPI app
app = FastAPI()

# Define the request/response body models
class FaceIdentificationRequest(BaseModel):
    image_base64: str

class FaceIdentificationResponse(BaseModel):
    name: str
    image_base64: str
    matching_image_base64: str
    confidence: int

class FaceSaveRequest(BaseModel):
    name: str
    image_base64: str

class FaceLabelRequest(BaseModel):
    face_image_url: str
    name: str

class FaceResponse(BaseModel):
    face_image_url: str
    name: str

class FaceImageResponse(BaseModel):
    face_image_url: str
    name: str
    image_base64: str

class FaceDeleteRequest(BaseModel):
    face_image_url: str

class FaceDeleteResponse(BaseModel):
    face_image_url: str
    name: str
    image_base64: str

@app.post("/identify_faces", description="Locates and identifies the faces in an image")
async def identify_faces(request: FaceIdentificationRequest):
    identified_faces = fr.recognize_faces_in_image(request.image_base64)
    response = []
    for face in identified_faces:
        response.append(FaceIdentificationResponse(
            name=face['name'],
            image_base64=face['image'],
            matching_image_base64=face['matching_image'],
            confidence=face['confidence']
        ))
    return response

@app.post("/save_face", description="Saves a face image for a person to the known faces database, or an unknown face if the name is not provided, or it is 'unknown'")
async def save_face(request: FaceSaveRequest):
    file_url = fr.save_face_image(request.image_base64, request.name)
    response = FaceResponse(
        face_image_url=file_url,
        name = request.name
    )
    return response

@app.get("/get_images", description="Retrieves all the face images for a specific person, or all the unknown faces if no name is provided.")
async def get_images(name: Optional[str] = None):
    images = fr.get_all_images(name)
    response = []
    for image in images:
        response.append(FaceImageResponse(
            face_image_url=image['face_image_url'],
            name=image['name'],
            image_base64=image['image_base64']
        ))
    return response

@app.post("/delete_face", description="Deletes a face image from the known or unknown faces database")
async def delete_face(request: FaceDeleteRequest):
    image = fr.delete_image(request.face_image_url)
    response = FaceDeleteResponse(
        face_image_url=image['face_image_url'],
        name=image['name'],
        image_base64=image['image_base64']
    )
    return response

@app.post("/label_face")
async def label_face(request: FaceLabelRequest, description="Labels an unknown face image with a name once they have been identified, or re-labels an existing person if they have been misidentified (as known or unknown if no name is provided)."):
    new_file_path = fr.label_image(request.face_image_url, request.name)
    response = FaceResponse(
        face_image_url=new_file_path,
        name=request.name
    )
    return response

if __name__ == "__main__":
    app.mount("/images", StaticFiles(directory=face_database_dir), name="images")
    uvicorn.run(app, host="0.0.0.0", port=8000)
