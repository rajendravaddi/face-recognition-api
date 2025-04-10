from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import numpy as np
import joblib

# Initialize model with optimizations
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(320, 320))  # Use CPU, smaller resolution

# Load classifier and encoder
classifier = joblib.load("cosface_model.joblib")
label_encoder = joblib.load("cosface_encoder.joblib")

api = FastAPI()

@api.post("/model-predict")
async def model_predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    img = np.array(image)

    faces = app.get(img)
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

