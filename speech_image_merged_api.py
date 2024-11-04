from fastapi import FastAPI, HTTPException, Header, File, UploadFile
from pydantic import BaseModel
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          AutoModelForImageClassification, AutoImageProcessor)
import torch
from typing import Optional, Union, List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Content Moderation API",
    description=
    "API for detecting hate speech and NSFW content in text and images",
    version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
TEXT_MODEL_PATH = "D:/content_safeguard_model_api/Hate Speech/hub/models--KoalaAI--OffensiveSpeechDetector/snapshots/e25a05698ea3405bcd0ed309f3bde50f722ba8ec"
IMAGE_MODEL_PATH = Path(
    r"D:\content_safeguard_model_api\Image Unsafe\models--MichalMlodawski--nsfw-image-detection-large\snapshots\631537d3bb871f59abb2769c1127359920a73640"
)

# Load text model
try:
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
    text_model = AutoModelForSequenceClassification.from_pretrained(
        TEXT_MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading text model: {str(e)}")
    raise

image_processor = None
image_model = None


def load_image_model():
    """Initialize the image classification model and processor"""
    global image_model, image_processor
    try:
        if not IMAGE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {IMAGE_MODEL_PATH}")

        logger.info("Loading image model and processor...")
        image_processor = AutoImageProcessor.from_pretrained(
            str(IMAGE_MODEL_PATH))
        image_model = AutoModelForImageClassification.from_pretrained(
            str(IMAGE_MODEL_PATH))
        image_model.eval()
        logger.info("Image model and processor loaded successfully")

    except Exception as e:
        logger.error(f"Error loading image model or processor: {str(e)}")
        raise RuntimeError(
            f"Failed to load image model and processor: {str(e)}")


load_image_model()


class InferenceRequest(BaseModel):
    inputs: Union[str, List[str]]


async def verify_token(authorization: Optional[str] = Header(None)):
    """Dummy token verification"""
    return "Any token accepted for testing"


@app.post("/predict")
async def predict(request: InferenceRequest,
                  token: str = Header(None, alias="Authorization")):
    await verify_token(token)

    try:
        inputs = request.inputs if isinstance(request.inputs,
                                              list) else [request.inputs]
        results = []
        for text in inputs:
            encoded = tokenizer(text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512)
            with torch.no_grad():
                outputs = text_model(**encoded)

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probabilities[0].tolist()
            label_id = torch.argmax(probabilities[0]).item()
            label = text_model.config.id2label[label_id]

            result = {"label": label, "score": probs[label_id]}
            results.append(result)

        return results[0] if isinstance(request.inputs, str) else results

    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error processing request: {str(e)}")


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=
            f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        image = Image.open(file.file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = image_model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            label_id = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][label_id].item()
            label = image_model.config.id2label[label_id]

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error processing image: {str(e)}")
    finally:
        file.file.close()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model_loaded": text_model is not None and image_model is not None
    }


if __name__ == "__main__":
    try:
        uvicorn.run(app,
                    host="127.0.0.1",
                    port=8000,
                    reload=True,
                    log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
