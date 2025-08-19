# app.py
import io, json, os, time
import numpy as np
from PIL import Image
from typing import Optional, List
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, confloat
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect

APP_VERSION = "0.1.0"
INPUT_SIZE = (32, 32)  # CIFAR-10
MAX_IMAGE_MB = 10

# ---- Model load ----
model = tf.keras.models.load_model("model.keras")
with open("labels.json","r") as f: LABELS = {int(k): v for k, v in json.load(f).items()}
infer = tf.function(model, autograph=False)

# ---- FastAPI ----
app = FastAPI(title="CIFAR10 FastAPI", version=APP_VERSION)
if os.getenv("ENABLE_CORS","0") in ("1","true","True"):
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictResponse(BaseModel):
    pred_id: int
    pred_label: str
    score: confloat(ge=0.0, le=1.0)
    time_ms: int
    model_version: Optional[str] = APP_VERSION

def load_image_rgb(raw: bytes) -> Image.Image:
    if len(raw) > MAX_IMAGE_MB*1024*1024:
        raise HTTPException(status_code=413, detail="Imagen demasiado grande")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen (jpg/png)")

def preprocess(img: Image.Image, wh=(32,32)) -> np.ndarray:
    w, h = wh
    x = np.asarray(img.resize((w,h)), dtype=np.float32)
    return x[None, ...]  # [1,H,W,3]

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION, "input_size": INPUT_SIZE}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg","image/png"):
        raise HTTPException(status_code=415, detail="Formato no admitido (usa jpg o png)")
    img = load_image_rgb(await file.read())
    x = preprocess(img, INPUT_SIZE)
    t0 = time.time()
    probs = infer(tf.constant(x), training=False).numpy()[0]  # [10]
    pid = int(probs.argmax())
    return PredictResponse(
        pred_id=pid,
        pred_label=LABELS.get(pid, str(pid)),
        score=float(probs[pid]),
        time_ms=int((time.time()-t0)*1000),
    )

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        raw = await f.read()
        img = load_image_rgb(raw)
        x = preprocess(img, INPUT_SIZE)
        probs = infer(tf.constant(x), training=False).numpy()[0]
        pid = int(probs.argmax())
        results.append({
            "filename": f.filename,
            "pred_id": pid,
            "pred_label": LABELS.get(pid, str(pid)),
            "score": float(probs[pid])
        })
    return {"results": results}

@app.websocket("/ws")
async def ws_predict(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_bytes()                  # bytes de una imagen
            img = load_image_rgb(raw)                       # valida/convierte a RGB
            x = preprocess(img, INPUT_SIZE)                 # (1,H,W,3) en [0,1]
            t0 = time.time()
            probs = infer(tf.constant(x), training=False).numpy()[0]
            pid = int(probs.argmax())
            resp = {
                "pred_id": pid,
                "pred_label": LABELS.get(pid, str(pid)),
                "score": float(probs[pid]),
                "time_ms": int((time.time() - t0) * 1000),
            }
            await ws.send_json(resp)
    except WebSocketDisconnect:
        pass