# Load the environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the logger (first things first)
import logging
import logging.config
import os
log_config_file = os.getenv('LOG_CONFIG_FILE', 'logging.ini')
logging.config.fileConfig(log_config_file)
logger = logging.getLogger(__name__)

from typing import List
from pydantic import BaseModel, Field
from typing import Optional
from fastapi import FastAPI, Request, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_403_FORBIDDEN
from starlette.responses import JSONResponse
import uvicorn

from watch.facial_recognition import FacialRecognition

# Facial Recognition Configuration
model = os.getenv('MODEL', 'default')  # "default" or "cnn" (cnn requires more GPU)
tolerance = float(os.getenv('TOLERANCE', '0.6'))  # Lower values make the recognition more strict, default is 0.6
face_database_dir = os.getenv('FACE_DATABASE_DIR', 'face_database')  # Where to store the known faces (auto created if it doesn't exist)

# Create an instance of the FacialRecognition class
fr = FacialRecognition(model, tolerance, face_database_dir)

# API configuration
api_port = int(os.getenv('API_PORT', '8000'))
api_host = os.getenv('API_HOST', 'localhost')
api_protocol = os.getenv('API_PROTOCOL', 'http')
api_root_path = os.getenv('API_ROOT_PATH', '')

# Set up the API documentation URLs
docs_swagger_url = os.getenv('DOCS_SWAGGER_URL')
docs_redoc_url = os.getenv('DOCS_REDOC_URL')
if docs_swagger_url is not None:
    logger.info(f"Swagger UI URL: {api_protocol}://{api_host}:{api_port}{api_root_path}{docs_swagger_url}")
if docs_redoc_url is not None:
    logger.info(f"ReDoc URL: {api_protocol}://{api_host}:{api_port}{api_root_path}{docs_redoc_url}")

# Create the FastAPI app
app = FastAPI(root_path=api_root_path, docs_url=docs_swagger_url, redoc_url=docs_redoc_url)

# mount the face database directory as a static directory
app.mount("/images", StaticFiles(directory=face_database_dir), name="images")
logger.info(f"Mounted face database directory `{face_database_dir}` at /images")

# Add CORS middleware headers
allowed_origins = os.getenv('ALLOWED_ORIGINS', '*')
origins = [origin.strip() for origin in allowed_origins.split(',')]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"Allowed origins: {origins}")

# Adding API key validation middleware
api_keys_file = os.getenv('API_KEYS_FILE')
valid_api_keys = []
if api_keys_file:
    logger.info(f"Loading API keys from file:{api_keys_file}")
    with open(api_keys_file, "r") as file:
        valid_api_keys = [line.strip() for line in file]
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
async def check_api_key(api_key_header: str = Security(api_key_header)):
    if api_keys_file and api_key_header not in valid_api_keys:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key_header

# Adding exception handling middleware
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    endpoint = request.url.path
    logger.error(f"An error occurred at endpoint {endpoint}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An error occurred: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": request.headers.get('Origin', ''),
            "Access-Control-Allow-Methods": request.headers.get('Access-Control-Request-Method', ''),
            "Access-Control-Allow-Headers": request.headers.get('Access-Control-Request-Headers', ''),
        },
    )

# Define the request/response body models
class FaceIdentificationRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image of the face to be identified")

class FaceIdentificationResponse(BaseModel):
    name: str = Field(..., description="Name of the identified person, or 'unknown' if the person is not recognized")
    image_base64: str = Field(..., description="Base64 encoded image of the identified face (cropped)")
    matching_image_base64: str = Field(..., description="Base64 encoded image of the matching face in the database")
    confidence: int = Field(..., description="Confidence level of the match expressed as a percentage (0-100)")

class FaceSaveRequest(BaseModel):
    name: str = Field(None, description="Name of the person in the image, or 'unknown' if the person is not known")
    image_base64: str = Field(..., description="Base64 encoded image of the face to be saved")

class FaceLabelRequest(BaseModel):
    face_image_url: str = Field(..., description="URL of the face image to be labeled (relative to the face database)")
    name: str = Field(None, description="Name of the person in the image, or 'unknown' if the person is not known")

class FaceResponse(BaseModel):
    face_image_url: str = Field(..., description="URL of the saved face image (relative to the face database)")
    name: str = Field(None, description="Name of the person in the image, or 'unknown' if the person is not known")

class FaceImageResponse(BaseModel):
    face_image_url: str = Field(..., description="URL of the face image (relative to the face database)")
    name: str = Field(None, description="Name of the person in the image, or 'unknown' if the person is not known")
    image_base64: str = Field(..., description="Base64 encoded image of the face")

class FaceDeleteRequest(BaseModel):
    face_image_url: str = Field(..., description="URL of the face image to be deleted (relative to the face database)")

class FaceDeleteResponse(BaseModel):
    face_image_url: str = Field(..., description="URL of the deleted face image (relative to the face database)")
    name: str = Field(None, description="Name of the person in the image, or 'unknown' if the person is not known")
    image_base64: str = Field(..., description="Base64 encoded image of the face")

@app.post("/identify_faces", 
          dependencies=[Depends(check_api_key)], 
          response_model=List[FaceIdentificationResponse], 
          description="Locates and identifies the faces in an image")
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

@app.post("/save_face", 
          dependencies=[Depends(check_api_key)], 
          response_model=FaceResponse,
          description="Saves a face image for a person to the known faces database, or an unknown face if the name is not provided, or it is 'unknown'")
async def save_face(request: FaceSaveRequest):
    file_url = fr.save_face_image(request.image_base64, request.name)
    response = FaceResponse(
        face_image_url=file_url,
        name = request.name
    )
    return response

@app.get("/get_images", 
         dependencies=[Depends(check_api_key)], 
         response_model=List[FaceImageResponse],
         description="Retrieves all the face images for a specific person, or all the unknown faces if no name is provided.")
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

@app.post("/delete_face", 
          dependencies=[Depends(check_api_key)], 
          response_model=FaceDeleteResponse,
          description="Deletes a face image from the known or unknown faces database")
async def delete_face(request: FaceDeleteRequest):
    image = fr.delete_image(request.face_image_url)
    response = FaceDeleteResponse(
        face_image_url=image['face_image_url'],
        name=image['name'],
        image_base64=image['image_base64']
    )
    return response

@app.post("/label_face", 
          dependencies=[Depends(check_api_key)], 
          response_model=FaceResponse,
          description="Labels an unknown face image with a name once they have been identified, or re-labels an existing person if they have been misidentified (as known or unknown if no name is provided).")
async def label_face(request: FaceLabelRequest):
    new_file_path = fr.label_image(request.face_image_url, request.name)
    response = FaceResponse(
        face_image_url=new_file_path,
        name=request.name
    )
    return response

if __name__ == "__main__":
    uvicorn.run(app, host=api_host, port=api_port, access_log=True, log_level="debug")
