from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

from app.orientation import analyse_orientation

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