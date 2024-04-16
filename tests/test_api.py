import base64
import re
import os
import shutil
import pytest

from dotenv import load_dotenv
load_dotenv("tests/.env")
face_database_dir = os.getenv('FACE_DATABASE_DIR', 'face_database')

from fastapi.testclient import TestClient
from watch.api import app

# Create a test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_app():
    # set up test data
    # run the test
    yield client

def test_identify_faces_unknown():
    # Prepare test data
    with open(os.path.join("tests","test_image.jpg"), "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    request_data = {
        "image_base64": image_base64
    }

    # Send a POST request to the endpoint
    response = client.post("/identify_faces", json=request_data)

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    response_data = response.json()
    assert isinstance(response_data, list)
    assert len(response_data) == 1

    # Check the response structure
    for face in response_data:
        assert "name" in face
        assert face["name"] == "unknown"
        assert "image_base64" in face
        assert "matching_image_base64" in face
        assert face["matching_image_base64"] == ""
        assert "confidence" in face
        assert face["confidence"] == 0

def test_identify_faces_known():
    # Prepare test data
    with open(os.path.join("tests","me.png"), "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    request_data = {
        "image_base64": image_base64
    }

    # Send a POST request to the endpoint
    response = client.post("/identify_faces", json=request_data)

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    response_data = response.json()
    assert isinstance(response_data, list)
    assert len(response_data) == 1

    # Check the response structure
    for face in response_data:
        assert "name" in face
        assert face["name"] == "Dagmar Timler"
        assert "image_base64" in face
        assert "matching_image_base64" in face
        assert "confidence" in face
        assert face["confidence"] == 50

def test_save_face():
    # Prepare test data
    with open(os.path.join("tests","test_image.jpg"), "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    request_data = {
        "image_base64": image_base64,
        "name": "John Doe"
    }

    # Send a POST request to the endpoint
    response = client.post("/save_face", json=request_data)

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    response_data = response.json()
    face_url = response_data["face_image_url"]
    assert face_url is not None
    assert re.search(r"John Doe/John Doe_\d+\.jpg", face_url) is not None
    assert response_data["name"] == "John Doe"

    # Cleanup
    shutil.rmtree(os.path.join(face_database_dir, "known", "John Doe"))

def test_label_face():
    # copy face from tests/test_image.jpg to /face_database/unkown/unknown_1.jpg
    unknown_dir = os.path.join(face_database_dir, "unknown")
    unknown_image_path = os.path.join(unknown_dir, "unknown_1.jpg")
    shutil.copy(os.path.join("tests","test_image.jpg"), unknown_image_path)

    # Prepare test data
    request_data = {
        "face_image_url": "unknown/unknown_1.jpg",
        "name": "Jane Doe"
    }

    # Send a POST request to the endpoint
    response = client.post("/label_face", json=request_data)

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    response_data = response.json()
    face_url = response_data["face_image_url"]
    assert face_url is not None
    assert re.search(r"Jane Doe/Jane Doe_\d+\.jpg", face_url) is not None
    assert response_data["name"] == "Jane Doe"
    assert os.path.exists(os.path.join(face_database_dir, face_url))
    assert not os.path.exists(unknown_image_path)

    # Cleanup
    shutil.rmtree(os.path.join(face_database_dir, "known", "Jane Doe"))

def test_delete_face():
    # copy face from tests/test_image.jpg to /face_database/unkown/unknown_1.jpg
    unknown_dir = os.path.join(face_database_dir, "unknown")
    unknown_image_path = os.path.join(unknown_dir, "unknown_1.jpg")
    shutil.copy(os.path.join("tests","test_image.jpg"), unknown_image_path)

    # Prepare test data
    request_data = {
        "face_image_url": "unknown/unknown_1.jpg",
    }

    # Send a POST request to the endpoint
    response = client.post("/delete_face", json=request_data)

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    response_data = response.json()
    face_url = response_data["face_image_url"]
    assert face_url is not None
    assert face_url == "unknown/unknown_1.jpg"
    assert response_data["name"] == "unknown"
    assert response_data["image_base64"] is not None

def test_get_images_without_name():
    # Send a GET request to the endpoint
    response = client.get("/get_images")

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    response_data = response.json()
    assert isinstance(response_data, list)
    assert len(response_data) == 1
    assert response_data[0]["name"] == "unknown"
    assert response_data[0]["face_image_url"] == "unknown/unknown_2.jpg"
    assert response_data[0]["image_base64"] is not None

def test_get_images_with_name_no_photos():
    # Prepare test data
    name = "Joe Soap"

    # Send a GET request to the endpoint
    response = client.get(f"/get_images?name={name}")

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    response_data = response.json()
    assert isinstance(response_data, list)
    assert len(response_data) == 0

def test_get_images_with_name_known():
    # Prepare test data
    name = "Dagmar Timler"

    # Send a GET request to the endpoint
    response = client.get(f"/get_images?name={name}")

    # Check the response status code
    assert response.status_code == 200

    # Check the response content
    response_data = response.json()
    assert isinstance(response_data, list)
    assert len(response_data) == 2
    assert response_data[0]["name"] == "Dagmar Timler"
    assert response_data[0]["face_image_url"] == "known/Dagmar Timler/Dagmar Timler_152848.jpg"
    assert response_data[0]["image_base64"] is not None
    assert response_data[1]["name"] == "Dagmar Timler"
    assert response_data[1]["face_image_url"] == "known/Dagmar Timler/Dagmar Timler_221849.jpg"
    assert response_data[1]["image_base64"] is not None

# Run the tests
if __name__ == "__main__":
    pytest.main()