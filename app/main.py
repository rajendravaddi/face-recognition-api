# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import numpy as np
import joblib

# Initialize face detector
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load classifier and label encoder
classifier = joblib.load("app/model/cosface_model.joblib")
label_encoder = joblib.load("app/model/cosface_encoder.joblib")

app = FastAPI()

@app.post("/model-predict")
async def model_predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    img = np.array(image)

    faces = face_app.get(img)
    results = []

    for face in faces:
        emb = face['embedding']
        emb = emb / norm(emb)
        dist, idx = classifier.kneighbors([emb], return_distance=True)
        threshold = 0.45
        if dist[0][0] > threshold:
            name = "Unknown"
        else:
            name = label_encoder.inverse_transform([classifier._y[idx[0][0]]])[0]

        results.append({
            "name": name,
            "bbox": face['bbox'].tolist(),
            "distance": float(dist[0][0])
        })

    return JSONResponse(content={"results": results})

