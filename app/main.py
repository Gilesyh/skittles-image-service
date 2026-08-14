from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

from app.orientation import analyse_orientation
from app.scorecard_registration import register_scorecard

app = FastAPI()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Skittles image service is running"
    }


@app.post("/estimate-orientation")
async def estimate_orientation(file: UploadFile = File(...)):
    # Read uploaded file into memory
    contents = await file.read()

    # Convert bytes into an OpenCV image
    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {
            "error": "Unable to decode image."
        }

    result = analyse_orientation(image)

    # Remove the OpenCV image before returning JSON
    result.pop("display", None)

    return result


@app.post("/register-scorecard")
async def register_scorecard_endpoint(
    image: UploadFile = File(...),
    template: UploadFile = File(...)
):
    # Read both uploaded files into memory
    image_contents = await image.read()
    template_contents = await template.read()

    # Decode the photographed image
    photographed_image = cv2.imdecode(
        np.frombuffer(image_contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    # Decode the clean template
    template_image = cv2.imdecode(
        np.frombuffer(template_contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if photographed_image is None:
        return {
            "error": "Unable to decode photographed image."
        }

    if template_image is None:
        return {
            "error": "Unable to decode template image."
        }

    result = register_scorecard(
        photographed_image,
        template_image
    )

    # Homography is a NumPy matrix and cannot be returned directly as JSON.
    result.pop("homography", None)

    return result