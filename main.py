from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Skittles image service is running"
    }